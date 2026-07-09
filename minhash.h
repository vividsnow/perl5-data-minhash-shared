/*
 * minhash.h -- Shared-memory MinHash sketch for Linux
 *
 * Jaccard-similarity estimation: keeps k "minimum hash" registers so that two
 * sketches, compared register by register, estimate the Jaccard similarity of
 * their underlying sets by the fraction of registers that agree, in a fixed
 * amount of memory. Each item is hashed once (XXH3-64) and mixed with each
 * register's index, so the k registers behave like k independent min-hashes and
 * every register keeps the minimum value it has ever seen. The registers live
 * in a shared mapping so several processes update one sketch; a write-preferring
 * futex rwlock with reader-slot dead-process recovery guards mutation. Two
 * sketches of equal size can be merged (element-wise minimum -> the sketch of
 * the union of the two sets).
 *
 * Layout: Header -> reader_slots[1024] -> registers[k*8]
 */

#ifndef MNH_H
#define MNH_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <pthread.h>

#define XXH_INLINE_ALL
#include "xxhash.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "minhash.h: requires little-endian architecture"
#endif


/* ================================================================
 * Constants
 * ================================================================ */

#define MNH_MAGIC        0x484E494DU  /* MinHash */
#define MNH_VERSION      1
#define MNH_ERR_BUFLEN   256
#ifndef MNH_READER_SLOTS
#define MNH_READER_SLOTS 1024         /* max concurrent reader processes for dead-process recovery */
#endif
#define MNH_MIN_K        1
#define MNH_MAX_K        0x1000000ULL   /* 2^24 registers cap */

#define MNH_ERR(fmt, ...) do { if (errbuf) snprintf(errbuf, MNH_ERR_BUFLEN, fmt, ##__VA_ARGS__); } while (0)

/* ================================================================
 * Structs
 * ================================================================ */

/* Per-process slot for dead-process recovery.  Each shared rwlock counter
 * (the main rwlock-reader count, rwlock_waiters, rwlock_writers_waiting)
 * is mirrored here so a wrlock timeout can attribute and reverse a dead
 * process's contribution instead of waiting for the slow per-op timeout
 * drain. */
typedef struct {
    uint32_t pid;            /* 0 = unclaimed */
    uint32_t subcount;       /* in-flight rdlock acquisitions for this process */
    uint32_t waiters_parked; /* contribution to hdr->rwlock_waiters         */
    uint32_t writers_parked; /* contribution to hdr->rwlock_writers_waiting */
} MnhReaderSlot;

struct MnhHeader {
    uint32_t magic, version;          /* 0,4 */
    uint32_t _pad0;                   /* 8 */
    uint32_t _pad1;                   /* 12 */
    uint64_t k;                       /* 16  number of min-hash registers */
    uint64_t _reserved0;              /* 24 */
    uint64_t capacity;                /* 32  == k (family stats parity) */
    uint64_t _reserved1;              /* 40 */
    uint64_t total_size;              /* 48 */
    uint64_t reader_slots_off;        /* 56 */
    uint64_t registers_off;           /* 64 */
    uint32_t rwlock;                  /* 72 */
    uint32_t rwlock_waiters;          /* 76 */
    uint32_t rwlock_writers_waiting;  /* 80 */
    uint32_t slotless_readers;  /* live readers holding the lock with no reader-slot (was padding) */
    uint64_t stat_ops;                /* 88 */
    uint8_t  _pad[160];               /* 96..255 */
};
typedef struct MnhHeader MnhHeader;

_Static_assert(sizeof(MnhHeader) == 256, "MnhHeader must be 256 bytes");

/* ---- Process-local handle ---- */

typedef struct MnhHandle {
    MnhHeader     *hdr;
    MnhReaderSlot *reader_slots;  /* MNH_READER_SLOTS entries */
    void         *base;          /* mmap base */
    uint64_t      registers_off;  /* validated register-array offset, cached: never re-read from the peer-writable header */
    size_t        mmap_size;
    char         *path;          /* backing file path (strdup'd) */
    int           backing_fd;    /* memfd or reopened-fd to close on destroy, -1 for file/anon */
    uint32_t      my_slot_idx;   /* UINT32_MAX if all slots taken (no recovery for this handle) */
    uint32_t      cached_pid;    /* getpid() cached at last slot claim */
    uint32_t      cached_fork_gen; /* mnh_fork_gen value at last slot claim */
    uint32_t slotless_held; /* rwlock read-locks held with no reader-slot */
} MnhHandle;

/* ================================================================
 * Futex-based write-preferring read-write lock
 * with reader-slot dead-process recovery
 * ================================================================ */

#define MNH_RWLOCK_SPIN_LIMIT 32
#define MNH_LOCK_TIMEOUT_SEC  2  /* FUTEX_WAIT timeout for stale lock detection */

static inline void mnh_rwlock_spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* Extract writer PID from rwlock value (lower 31 bits when write-locked). */
#define MNH_RWLOCK_WRITER_BIT 0x80000000U
#define MNH_RWLOCK_PID_MASK   0x7FFFFFFFU
#define MNH_RWLOCK_WR(pid)    (MNH_RWLOCK_WRITER_BIT | ((uint32_t)(pid) & MNH_RWLOCK_PID_MASK))

/* Check if a PID is alive. Returns 1 if alive or unknown, 0 if definitely dead. */
/* Liveness via kill(pid,0). NOTE: cannot detect PID reuse -- if a dead
 * lock-holder's PID is recycled to an unrelated live process before recovery
 * runs, this reports "alive" and that slot's orphaned contribution is not
 * reclaimed until the recycled process exits. Robust detection would require
 * a per-slot process-start-time epoch (a header-layout/version change).
 * Documented under "Crash Safety" in the POD. */
static inline int mnh_pid_alive(uint32_t pid) {
    if (pid == 0) return 1; /* no owner recorded, assume alive */
    return !(kill((pid_t)pid, 0) == -1 && errno == ESRCH);
}

/* Force-recover a stale write lock left by a dead process.
 * CAS to OUR pid to hold the lock while fixing shared state, then release.
 * Using our pid (not a bare WRITER_BIT sentinel) means a subsequent
 * recovering process can detect and re-recover if we crash mid-recovery. */
static inline void mnh_recover_stale_lock(MnhHandle *h, uint32_t observed_rwlock) {
    MnhHeader *hdr = h->hdr;
    uint32_t mypid = MNH_RWLOCK_WR((uint32_t)getpid());
    if (!__atomic_compare_exchange_n(&hdr->rwlock, &observed_rwlock,
            mypid, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return;
    /* We now hold the write lock as mypid.  No additional shared state needs
     * repair here (this module has no seqlock); just release the lock. */
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static const struct timespec mnh_lock_timeout = { MNH_LOCK_TIMEOUT_SEC, 0 };

/* Process-global fork-generation counter.  Incremented in the pthread_atfork
 * child callback so every open handle detects a fork transition on the next
 * lock call without paying a getpid() syscall on the hot path. */
static uint32_t mnh_fork_gen = 1;
static pthread_once_t mnh_atfork_once = PTHREAD_ONCE_INIT;
static void mnh_on_fork_child(void) {
    __atomic_add_fetch(&mnh_fork_gen, 1, __ATOMIC_RELAXED);
}
static void mnh_atfork_init(void) {
    pthread_atfork(NULL, NULL, mnh_on_fork_child);
}

/* Ensure this process owns a reader slot.  Called from the lock helpers so
 * that fork()'d children pick up their own slot lazily instead of sharing
 * the parent's.  Hot-path is a single relaxed load + compare; only on a
 * fork-generation mismatch do we touch getpid() and scan slots. */
static inline void mnh_claim_reader_slot(MnhHandle *h) {
    uint32_t cur_gen = __atomic_load_n(&mnh_fork_gen, __ATOMIC_RELAXED);
    if (__builtin_expect(cur_gen == h->cached_fork_gen && h->my_slot_idx != UINT32_MAX, 1))
        return;
    /* Cold path -- register the atfork hook once per process, then claim. */
    pthread_once(&mnh_atfork_once, mnh_atfork_init);
    /* Re-read after pthread_once: mnh_on_fork_child may have bumped it. */
    cur_gen = __atomic_load_n(&mnh_fork_gen, __ATOMIC_RELAXED);
    uint32_t now_pid = (uint32_t)getpid();
    h->cached_pid = now_pid;
    if (cur_gen != h->cached_fork_gen) h->slotless_held = 0;  /* fork: child holds none of the parent's slotless read locks */
    h->cached_fork_gen = cur_gen;
    h->my_slot_idx = UINT32_MAX;
    uint32_t start = now_pid % MNH_READER_SLOTS;
    for (uint32_t i = 0; i < MNH_READER_SLOTS; i++) {
        uint32_t s = (start + i) % MNH_READER_SLOTS;
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(&h->reader_slots[s].pid,
                &expected, now_pid, 0,
                __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            /* Zero all mirror fields, not just subcount: a SIGKILL'd
             * predecessor may have left waiters_parked/writers_parked
             * non-zero, and mnh_recover_dead_readers won't drain them
             * once we own the slot (the CAS expects the dead PID). */
            __atomic_store_n(&h->reader_slots[s].subcount, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].waiters_parked, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].writers_parked, 0, __ATOMIC_RELAXED);
            h->my_slot_idx = s;
            return;
        }
    }
    /* Table full -- leave my_slot_idx = UINT32_MAX so we silently skip
     * tracking for this handle (lock still works; just no recovery). */
}

/* Atomically subtract `sub` from a counter, capped at 0 (never underflows). */
static inline void mnh_atomic_sub_cap(uint32_t *p, uint32_t sub) {
    if (!sub) return;
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t want = (cur > sub) ? cur - sub : 0;
        if (__atomic_compare_exchange_n(p, &cur, want,
                1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

/* Try to claim a dead slot (CAS pid -> 0) and drain its parked-waiter
 * contributions back to the global counters.  A no-op if the slot was stolen
 * by another recoverer or had no waiter contribution to drain.
 *
 * Note: subcount/waiters_parked/writers_parked are NOT zeroed here.
 * Between our CAS and a follow-up store, a new process could claim the
 * slot and start populating these fields -- our stores would clobber its
 * state.  mnh_claim_reader_slot zeros all three on every claim, so
 * leaving stale values is harmless. */
static inline void mnh_drain_dead_slot(MnhHandle *h, uint32_t i, uint32_t pid) {
    MnhHeader *hdr = h->hdr;
    uint32_t expected = pid;
    /* ACQ_REL on success: RELEASE publishes pid=0 to other observers;
     * ACQUIRE syncs us with prior writes from the dead process to
     * waiters_parked/writers_parked.  On weakly-ordered archs (aarch64)
     * a plain RELAXED load before the CAS could miss those writes;
     * loading them after the CAS keeps them inside the acquire window. */
    if (!__atomic_compare_exchange_n(&h->reader_slots[i].pid, &expected, 0,
            0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    uint32_t wp    = __atomic_load_n(&h->reader_slots[i].waiters_parked, __ATOMIC_RELAXED);
    uint32_t writp = __atomic_load_n(&h->reader_slots[i].writers_parked, __ATOMIC_RELAXED);
    if (wp)    mnh_atomic_sub_cap(&hdr->rwlock_waiters, wp);
    if (writp) mnh_atomic_sub_cap(&hdr->rwlock_writers_waiting, writp);
}

/* Scan reader slots for dead-process recovery.
 *
 * For each dead PID with non-zero contributions to the shared rwlock,
 * rwlock_waiters, or rwlock_writers_waiting counters, drain its share back
 * out so live processes don't have to wait for the slow per-op timeout
 * decrement to drain it for them.
 *
 * For the main rwlock counter we use the "no live reader holds -> force-
 * reset to 0" trick (precise) because per-process attribution of the
 * subcount is racy across the inc-counter-then-inc-subcount window. */
static inline void mnh_recover_dead_readers(MnhHandle *h) {
    if (!h->reader_slots) return;
    MnhHeader *hdr = h->hdr;
    int any_live_reader = 0;
    int found_dead_reader = 0;

    /* Pass 1: classify slots.  Slots with dead pid and sc == 0 (no rwlock
     * contribution to lose) are wiped immediately to free the slot for
     * future claimants and drain any orphan parked-waiter counters.  Slots
     * with dead pid and sc > 0 are left intact in this pass: if force-
     * reset cannot fire (because a live reader is concurrently present),
     * wiping the dead slot would lose the only record of its orphan
     * rwlock contribution and strand writers permanently once the live
     * reader releases. */
    for (uint32_t i = 0; i < MNH_READER_SLOTS; i++) {
        uint32_t pid = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
        if (pid == 0) continue;
        uint32_t sc = __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED);
        if (mnh_pid_alive(pid)) {
            if (sc > 0) any_live_reader = 1;
            continue;
        }
        if (sc > 0) { found_dead_reader = 1; continue; }
        mnh_drain_dead_slot(h, i, pid);
    }

    /* Pass 2: only if force-reset will fire.  Issue the rwlock force-
     * reset CAS FIRST, while the window since pass 1's last scan is
     * still narrow (a handful of instructions, as in the original
     * single-pass code).  A new reader that started rdlock between
     * pass 1's scan and the CAS will either:
     *   (a) have already CAS'd rwlock from cur to cur+1 -- our CAS then
     *       fails (cur mismatched), recovery yields and a future
     *       cycle retries; or
     *   (b) be still in the subcount-bump phase -- our CAS sees the
     *       stale cur and resets to 0; the new reader's subsequent CAS
     *       rwlock(0 -> 1) succeeds cleanly.
     * Only after the CAS resolves do we wipe the deferred dead slots,
     * keeping that work outside the race-sensitive window. */
    /* A live reader with no slot (table was full) is invisible to the scan
     * above but still holds a +1 in the lock word; never force-reset under it. */
    if (__atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0)
        any_live_reader = 1;
    if (found_dead_reader && !any_live_reader) {
        /* ACQUIRE: a late reader's subcount++ (before its rwlock CAS) is then visible below. */
        uint32_t cur = __atomic_load_n(&hdr->rwlock, __ATOMIC_ACQUIRE);
        int drain_ok = 1;   /* keep dead slots if the reset doesn't fire */
        if (cur > 0 && cur < MNH_RWLOCK_WRITER_BIT) {
            /* Re-scan for a live reader (fail-safe: only suppresses a reset). */
            int live_now = __atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0;
            for (uint32_t i = 0; !live_now && i < MNH_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p && mnh_pid_alive(p) &&
                    __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED) > 0)
                    live_now = 1;
            }
            if (live_now) {
                drain_ok = 0;
            } else if (__atomic_compare_exchange_n(&hdr->rwlock, &cur, 0,
                    0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
                    syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
            } else {
                drain_ok = 0;   /* rwlock changed under us -- shares may still be live */
            }
        }
        if (drain_ok) {
            for (uint32_t i = 0; i < MNH_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p == 0 || mnh_pid_alive(p)) continue;
                mnh_drain_dead_slot(h, i, p);
            }
        }
    }
}

/* Inspect the lock word after a futex-wait timeout.  If a dead writer
 * holds it, force-recover the lock.  Otherwise drain dead readers' shares
 * of the rwlock/waiter counters.  Called from rdlock and wrlock ETIMEDOUT
 * branches -- identical recovery logic in both. */
static inline void mnh_recover_after_timeout(MnhHandle *h) {
    MnhHeader *hdr = h->hdr;
    uint32_t val = __atomic_load_n(&hdr->rwlock, __ATOMIC_RELAXED);
    if (val >= MNH_RWLOCK_WRITER_BIT) {
        uint32_t pid = val & MNH_RWLOCK_PID_MASK;
        if (!mnh_pid_alive(pid))
            mnh_recover_stale_lock(h, val);
    } else {
        mnh_recover_dead_readers(h);
    }
}

/* Park/unpark helpers: bump the global waiter counters together with this
 * process's mirrored slot counters so a wrlock-timeout recovery scan can
 * attribute and reverse a dead PID's contribution.  Kept paired to make
 * accidental drift between global and per-slot counts impossible. */
static inline void mnh_park_reader(MnhHandle *h) {
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
}
static inline void mnh_unpark_reader(MnhHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
}
static inline void mnh_park_writer(MnhHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
}
static inline void mnh_unpark_writer(MnhHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
}

/* Reader accounting: a reader mirrors its +1 in the lock word so dead-reader
 * recovery can see it. A slotted reader uses its slot subcount; a reader that
 * could not claim a slot (table full) uses the global hdr->slotless_readers,
 * so recovery's force-reset never fires out from under it. leave() peels
 * slotless first so a later slot claim cannot misattribute the decrement. */
static inline void mnh_reader_enter(MnhHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
        h->slotless_held++;
    }
}
static inline void mnh_reader_leave(MnhHandle *h) {
    if (h->slotless_held > 0) {
        h->slotless_held--;
        __atomic_sub_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
    } else if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    }
}

static inline void mnh_rwlock_rdlock(MnhHandle *h) {
    mnh_claim_reader_slot(h);
    MnhHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    uint32_t *writers_waiting = &hdr->rwlock_writers_waiting;
    /* Claim subcount BEFORE bumping the shared rwlock counter.  This way
     * a concurrent writer-side recovery scan that sees our PID alive with
     * subcount > 0 will (correctly) defer force-reset, even while we are
     * still spinning trying to win the rwlock CAS.  Without this, a reader
     * killed between rwlock CAS-success and subcount++ would let recovery
     * force-reset rwlock to 0 underneath us, causing a UINT32_MAX wrap on
     * our eventual rdunlock dec. */
    mnh_reader_enter(h);
    for (int spin = 0; ; spin++) {
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Write-preferring: when lock is free (cur==0) and writers are
         * waiting, yield to let the writer acquire. When readers are
         * already active (cur>=1), new readers may join freely. */
        if (cur > 0 && cur < MNH_RWLOCK_WRITER_BIT) {
            if (__atomic_compare_exchange_n(lock, &cur, cur + 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        } else if (cur == 0 && !__atomic_load_n(writers_waiting, __ATOMIC_RELAXED)) {
            if (__atomic_compare_exchange_n(lock, &cur, 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        }
        if (__builtin_expect(spin < MNH_RWLOCK_SPIN_LIMIT, 1)) {
            mnh_rwlock_spin_pause();
            continue;
        }
        mnh_park_reader(h);
        cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Sleep when write-locked OR when yielding to waiting writers */
        if (cur >= MNH_RWLOCK_WRITER_BIT || cur == 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &mnh_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                mnh_unpark_reader(h);
                mnh_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        mnh_unpark_reader(h);
        spin = 0;
    }
}

static inline void mnh_rwlock_rdunlock(MnhHandle *h) {
    MnhHeader *hdr = h->hdr;
    /* Release the shared counter BEFORE dropping our subcount so that
     * "any live PID with subcount > 0" is a reliable in-flight indicator
     * for the writer-side recovery scan.  Inverting these would create a
     * window where we still own a unit of rwlock but our slot subcount is
     * 0, letting recovery force-reset rwlock underneath us. */
    uint32_t after = __atomic_sub_fetch(&hdr->rwlock, 1, __ATOMIC_RELEASE);
    mnh_reader_leave(h);
    if (after == 0 && __atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static inline void mnh_rwlock_wrlock(MnhHandle *h) {
    mnh_claim_reader_slot(h);  /* refresh cached_pid across fork */
    MnhHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    /* Encode PID in the rwlock word itself (0x80000000 | pid) to eliminate
     * any crash window between acquiring the lock and storing the owner. */
    uint32_t mypid = MNH_RWLOCK_WR(h->cached_pid);
    for (int spin = 0; ; spin++) {
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(lock, &expected, mypid,
                1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            return;
        if (__builtin_expect(spin < MNH_RWLOCK_SPIN_LIMIT, 1)) {
            mnh_rwlock_spin_pause();
            continue;
        }
        mnh_park_writer(h);
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        if (cur != 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &mnh_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                mnh_unpark_writer(h);
                mnh_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        mnh_unpark_writer(h);
        spin = 0;
    }
}

static inline void mnh_rwlock_wrunlock(MnhHandle *h) {
    MnhHeader *hdr = h->hdr;
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

/* ================================================================
 * Layout math + create / open / destroy
 *
 * Layout: Header -> reader_slots[1024] -> registers[k*8]
 * ================================================================ */

/* Single source of truth for the mmap region layout offsets. */
typedef struct { uint64_t reader_slots, registers; } MnhLayout;

static inline MnhLayout mnh_layout(void) {
    MnhLayout L;
    L.reader_slots = sizeof(MnhHeader);
    L.registers    = L.reader_slots + (uint64_t)MNH_READER_SLOTS * sizeof(MnhReaderSlot);
    L.registers    = (L.registers + 7) & ~(uint64_t)7;   /* 8-byte align the uint64 registers */
    return L;
}

static inline uint64_t mnh_total_size(uint64_t k) {
    MnhLayout L = mnh_layout();
    return L.registers + k * sizeof(uint64_t);
}

static inline void mnh_init_header(void *base, uint64_t k, uint64_t total) {
    MnhLayout L = mnh_layout();
    MnhHeader *hdr = (MnhHeader *)base;
    /* zero the header + reader-slot region, then fill the registers with the
       empty sentinel UINT64_MAX (a min-update replaces it with any real hash) */
    memset(base, 0, (size_t)L.registers);
    hdr->magic            = MNH_MAGIC;
    hdr->version          = MNH_VERSION;
    hdr->k                = k;
    hdr->capacity         = k;
    hdr->total_size       = total;
    hdr->reader_slots_off = L.reader_slots;
    hdr->registers_off    = L.registers;
    memset((char *)base + L.registers, 0xFF, (size_t)(k * sizeof(uint64_t)));  /* registers = UINT64_MAX */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline uint64_t *mnh_registers(MnhHandle *h) {
    return (uint64_t *)((char *)h->base + h->registers_off);
}

/* Layer B trusted bound: number of uint64 registers guaranteed within the real
 * mapping.  Derived from the process-local mmap_size and the SAME registers_off
 * mnh_registers() uses, so a peer that corrupts hdr->k / registers_off after
 * attach-time validation can never drive an access outside the mapping.  Equals
 * k for a valid sketch, so every clamp below it is a never-taken branch. */
static inline uint64_t mnh_reg_max(MnhHandle *h) {
    uint64_t off = h->registers_off;
    if (off >= h->mmap_size) return 0;
    return (h->mmap_size - off) / sizeof(uint64_t);
}

static inline MnhHandle *mnh_setup(void *base, size_t map_size,
                                 const char *path, int backing_fd) {
    MnhHeader *hdr = (MnhHeader *)base;
    MnhHandle *h = (MnhHandle *)calloc(1, sizeof(MnhHandle));
    if (!h) {
        munmap(base, map_size);
        if (backing_fd >= 0) close(backing_fd);
        return NULL;
    }
    h->hdr          = hdr;
    h->base         = base;
    h->reader_slots = (MnhReaderSlot *)((uint8_t *)base + hdr->reader_slots_off);
    h->registers_off = hdr->registers_off;   /* single validated read; bound and pointer stay consistent */
    h->mmap_size    = map_size;
    h->path         = path ? strdup(path) : NULL;
    h->backing_fd   = backing_fd;
    h->my_slot_idx  = UINT32_MAX;
    return h;
}

/* Validate a mapped header (shared by mnh_create reopen and mnh_open_fd). */
static inline int mnh_validate_header(const MnhHeader *hdr, uint64_t file_size) {
    if (hdr->magic != MNH_MAGIC) return 0;
    if (hdr->version != MNH_VERSION) return 0;
    if (hdr->k < MNH_MIN_K || hdr->k > MNH_MAX_K) return 0;
    if (hdr->capacity != hdr->k) return 0;
    if (hdr->total_size != file_size) return 0;
    if (hdr->total_size != mnh_total_size(hdr->k)) return 0;
    MnhLayout L = mnh_layout();
    if (hdr->reader_slots_off != L.reader_slots) return 0;
    if (hdr->registers_off != L.registers) return 0;
    return 1;
}

/* validate the requested number of registers k */
static int mnh_validate_args(uint64_t k, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    if (k < MNH_MIN_K || k > MNH_MAX_K) { MNH_ERR("number of registers must be between 1 and 2^24"); return 0; }
    return 1;
}

/* Securely obtain a fd for a path-backed segment: create it exclusively
 * (O_CREAT|O_EXCL|O_NOFOLLOW at `mode`, default 0600 = owner-only), or, if it
 * already exists, attach to it (O_RDWR|O_NOFOLLOW, no O_CREAT). O_EXCL blocks a
 * pre-seeded or hard-linked file and O_NOFOLLOW a symlink swap, so a local
 * attacker can no longer redirect or poison the backing store through the path.
 * Cross-user sharing is opt-in via a wider `mode` (e.g. 0660); the caller still
 * validates the file's contents via mnh_validate_header. */
static int mnh_secure_open(const char *path, mode_t mode, char *errbuf) {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = open(path, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, mode);
        if (fd >= 0) { (void)fchmod(fd, mode); return fd; }   /* exact mode: umask narrowed the O_EXCL create */
        if (errno != EEXIST) { MNH_ERR("create %s: %s", path, strerror(errno)); return -1; }
        fd = open(path, O_RDWR|O_NOFOLLOW|O_CLOEXEC);
        if (fd >= 0) return fd;
        if (errno == ENOENT) continue;   /* creator unlinked between our two opens; retry */
        MNH_ERR("open %s: %s", path, strerror(errno));  /* ELOOP => symlink rejected */
        return -1;
    }
    MNH_ERR("open %s: create/attach kept racing", path);
    return -1;
}

static MnhHandle *mnh_create(const char *path, uint64_t k, mode_t mode, char *errbuf) {
    if (!mnh_validate_args(k, errbuf)) return NULL;

    uint64_t total = mnh_total_size(k);
    int anonymous = (path == NULL);
    int fd = -1;
    size_t map_size;
    void *base;

    if (anonymous) {
        map_size = (size_t)total;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) { MNH_ERR("mmap: %s", strerror(errno)); return NULL; }
    } else {
        fd = mnh_secure_open(path, mode, errbuf);
        if (fd < 0) return NULL;
        if (flock(fd, LOCK_EX) < 0) { MNH_ERR("flock: %s", strerror(errno)); close(fd); return NULL; }
        struct stat st;
        if (fstat(fd, &st) < 0) { MNH_ERR("fstat: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        int is_new = (st.st_size == 0);
        if (!is_new && (uint64_t)st.st_size < sizeof(MnhHeader)) {
            MNH_ERR("%s: file too small (%lld)", path, (long long)st.st_size);
            flock(fd, LOCK_UN); close(fd); return NULL;
        }
        if (is_new && ftruncate(fd, (off_t)total) < 0) {
            MNH_ERR("ftruncate: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL;
        }
        map_size = is_new ? (size_t)total : (size_t)st.st_size;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { MNH_ERR("mmap: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        if (!is_new) {
            if (!mnh_validate_header((MnhHeader *)base, (uint64_t)st.st_size)) {
                MNH_ERR("invalid MinHash sketch file"); munmap(base, map_size); flock(fd, LOCK_UN); close(fd); return NULL;
            }
            flock(fd, LOCK_UN); close(fd);
            return mnh_setup(base, map_size, path, -1);
        }
    }
    mnh_init_header(base, k, total);
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
    return mnh_setup(base, map_size, path, -1);
}

static MnhHandle *mnh_create_memfd(const char *name, uint64_t k, char *errbuf) {
    if (!mnh_validate_args(k, errbuf)) return NULL;

    uint64_t total = mnh_total_size(k);
    int fd = memfd_create(name ? name : "minhash", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) { MNH_ERR("memfd_create: %s", strerror(errno)); return NULL; }
    if (ftruncate(fd, (off_t)total) < 0) {
        MNH_ERR("ftruncate: %s", strerror(errno)); close(fd); return NULL;
    }
    (void)fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    void *base = mmap(NULL, (size_t)total, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { MNH_ERR("mmap: %s", strerror(errno)); close(fd); return NULL; }
    mnh_init_header(base, k, total);
    return mnh_setup(base, (size_t)total, NULL, fd);
}

static MnhHandle *mnh_open_fd(int fd, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    struct stat st;
    if (fstat(fd, &st) < 0) { MNH_ERR("fstat: %s", strerror(errno)); return NULL; }
    if ((uint64_t)st.st_size < sizeof(MnhHeader)) { MNH_ERR("too small"); return NULL; }
    size_t ms = (size_t)st.st_size;
    void *base = mmap(NULL, ms, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { MNH_ERR("mmap: %s", strerror(errno)); return NULL; }
    if (!mnh_validate_header((MnhHeader *)base, (uint64_t)st.st_size)) {
        MNH_ERR("invalid MinHash sketch table"); munmap(base, ms); return NULL;
    }
    int myfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (myfd < 0) { MNH_ERR("fcntl: %s", strerror(errno)); munmap(base, ms); return NULL; }
    return mnh_setup(base, ms, NULL, myfd);
}

static void mnh_destroy(MnhHandle *h) {
    if (!h) return;
    /* Release our reader slot on clean teardown (else short-lived-reader churn
     * exhausts the slot table); skip if a lock is still held (subcount>0). */
    if (h->reader_slots && h->my_slot_idx != UINT32_MAX && h->cached_pid &&
        h->cached_fork_gen == __atomic_load_n(&mnh_fork_gen, __ATOMIC_RELAXED) &&
        __atomic_load_n(&h->reader_slots[h->my_slot_idx].subcount, __ATOMIC_ACQUIRE) == 0) {
        uint32_t expected = h->cached_pid;
        __atomic_compare_exchange_n(&h->reader_slots[h->my_slot_idx].pid,
                &expected, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    if (h->backing_fd >= 0) close(h->backing_fd);
    if (h->base) munmap(h->base, h->mmap_size);
    free(h->path);
    free(h);
}

static inline int mnh_msync(MnhHandle *h) {
    if (!h || !h->base) return 0;
    return msync(h->base, h->mmap_size, MS_SYNC);
}

/* ================================================================
 * MinHash operations (callers hold the lock).  Each of the k registers
 * holds the minimum hash value seen for an independent hash function; an
 * item is hashed once, then mixed with each register's index so the k
 * registers behave like k independent min-hashes.  The Jaccard similarity
 * of two sets is estimated by the fraction of registers that agree between
 * their sketches.
 * ================================================================ */

/* mix a base hash with register index j into an independent 64-bit value
 * (a splitmix64-style finalizer keyed by j) */
static inline uint64_t mnh_mix(uint64_t base, uint64_t j) {
    uint64_t x = base + (j + 1) * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/* fold one item into the sketch: for each register keep the smaller of its
 * current value and this item's mixed hash.  Returns 1 if any register was
 * lowered (the item changed the sketch), else 0. */
static int mnh_add_locked(MnhHandle *h, const void *item, size_t len) {
    uint64_t base = XXH3_64bits(item, len);
    uint64_t k = h->hdr->k;
    uint64_t kmax = mnh_reg_max(h);        /* Layer B: clamp to the mapping */
    if (k > kmax) k = kmax;
    uint64_t *reg = mnh_registers(h);
    int changed = 0;
    for (uint64_t j = 0; j < k; j++) {
        uint64_t v = mnh_mix(base, j);
        if (v < reg[j]) { reg[j] = v; changed = 1; }
    }
    return changed;
}

/* count registers that agree between this sketch and a snapshot of another's
 * registers; the Jaccard estimate is agree / k.  other_k is how many
 * registers the snapshot buffer actually holds. */
static uint64_t mnh_agree_locked(MnhHandle *h, const uint64_t *other, uint64_t other_k) {
    uint64_t k = h->hdr->k;
    uint64_t kmax = mnh_reg_max(h);        /* Layer B: clamp to the mapping */
    if (k > kmax) k = kmax;
    if (k > other_k) k = other_k;          /* ...and to the snapshot buffer */
    uint64_t *reg = mnh_registers(h);
    uint64_t agree = 0;
    for (uint64_t j = 0; j < k; j++)
        if (reg[j] == other[j]) agree++;
    return agree;
}

/* number of registers that have taken a value (differ from the UINT64_MAX
 * empty sentinel); 0 means the sketch is empty. (caller holds a lock) */
static uint64_t mnh_filled_locked(MnhHandle *h) {
    uint64_t k = h->hdr->k;
    uint64_t kmax = mnh_reg_max(h);        /* Layer B: clamp to the mapping */
    if (k > kmax) k = kmax;
    uint64_t *reg = mnh_registers(h);
    uint64_t filled = 0;
    for (uint64_t j = 0; j < k; j++)
        if (reg[j] != UINT64_MAX) filled++;
    return filled;
}

/* merge another sketch's registers into this one: element-wise minimum, so
 * the result is the min-hash sketch of the union of the two input sets.
 * src_k is the number of registers the src buffer actually holds. */
static void mnh_merge_locked(MnhHandle *dst, const uint64_t *src, uint64_t src_k) {
    uint64_t k = dst->hdr->k;
    uint64_t kmax = mnh_reg_max(dst);      /* Layer B: clamp writes to dst mapping */
    if (k > kmax) k = kmax;
    if (k > src_k) k = src_k;              /* ...and reads to the src buffer */
    uint64_t *reg = mnh_registers(dst);
    for (uint64_t j = 0; j < k; j++)
        if (src[j] < reg[j]) reg[j] = src[j];
}

/* reset every register to the empty sentinel (caller holds the write lock) */
static inline void mnh_clear_locked(MnhHandle *h) {
    uint64_t k = h->hdr->k;
    uint64_t kmax = mnh_reg_max(h);        /* Layer B: clamp memset to the mapping */
    if (k > kmax) k = kmax;
    memset(mnh_registers(h), 0xFF, (size_t)(k * sizeof(uint64_t)));
}

#endif /* MNH_H */
