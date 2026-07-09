#define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#include "ppport.h"
#include "minhash.h"

#define EXTRACT(sv) \
    if (!sv_isobject(sv) || !sv_derived_from(sv, "Data::MinHash::Shared")) \
        croak("Expected a Data::MinHash::Shared object"); \
    MnhHandle *h = INT2PTR(MnhHandle*, SvIV(SvRV(sv))); \
    if (!h) croak("Attempted to use a destroyed Data::MinHash::Shared object")

#define MAKE_OBJ(class, handle) \
    SV *obj = newSViv(PTR2IV(handle)); \
    SV *ref = newRV_noinc(obj); \
    sv_bless(ref, gv_stashpv(class, GV_ADD)); \
    RETVAL = ref

MODULE = Data::MinHash::Shared  PACKAGE = Data::MinHash::Shared

PROTOTYPES: DISABLE

SV *
new(class, path = &PL_sv_undef, k = 0, ...)
    const char *class
    SV *path
    UV k
  PREINIT:
    char errbuf[MNH_ERR_BUFLEN];
  CODE:
    const char *p = (SvGETMAGIC(path), SvOK(path)) ? SvPV_nolen(path) : NULL;
    if (k < 1)
        croak("Data::MinHash::Shared->new: number of registers must be >= 1");
    /* Optional 4th arg: file mode for a newly-created file-backed segment
     * (default 0600, owner-only). Pass e.g. 0660 to opt into cross-user
     * sharing. Ignored for anonymous segments and existing files. */
    mode_t mode = (items > 3 && (SvGETMAGIC(ST(3)), SvOK(ST(3)))) ? (mode_t)SvUV(ST(3)) : 0600;
    MnhHandle *h = mnh_create(p, (uint64_t)k, mode, errbuf);
    if (!h) croak("Data::MinHash::Shared->new: %s", errbuf);
    MAKE_OBJ(class, h);
  OUTPUT:
    RETVAL

SV *
new_memfd(class, name = &PL_sv_undef, k = 0)
    const char *class
    SV *name
    UV k
  PREINIT:
    char errbuf[MNH_ERR_BUFLEN];
  CODE:
    const char *nm = (SvGETMAGIC(name), SvOK(name)) ? SvPV_nolen(name) : NULL;   /* undef -> default label */
    if (k < 1)
        croak("Data::MinHash::Shared->new_memfd: number of registers must be >= 1");
    MnhHandle *h = mnh_create_memfd(nm, (uint64_t)k, errbuf);
    if (!h) croak("Data::MinHash::Shared->new_memfd: %s", errbuf);
    MAKE_OBJ(class, h);
  OUTPUT:
    RETVAL

SV *
new_from_fd(class, fd)
    const char *class
    int fd
  PREINIT:
    char errbuf[MNH_ERR_BUFLEN];
  CODE:
    MnhHandle *h = mnh_open_fd(fd, errbuf);
    if (!h) croak("Data::MinHash::Shared->new_from_fd: %s", errbuf);
    MAKE_OBJ(class, h);
  OUTPUT:
    RETVAL

void
DESTROY(self)
    SV *self
  CODE:
    if (sv_isobject(self) && sv_derived_from(self, "Data::MinHash::Shared")) {
        MnhHandle *h = INT2PTR(MnhHandle*, SvIV(SvRV(self)));
        if (h) { sv_setiv(SvRV(self), 0); mnh_destroy(h); }   /* null first: activates EXTRACT's use-after-destroy croak + makes a double DESTROY a no-op */
    }

int
add(self, item)
    SV *self
    SV *item
  PREINIT:
    EXTRACT(self);
    STRLEN n;
    const char *s;
  CODE:
    s = SvPVbyte(item, n);                 /* may croak (wide char) -- BEFORE the lock */
    mnh_rwlock_wrlock(h);
    RETVAL = mnh_add_locked(h, s, n);
    __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);
    mnh_rwlock_wrunlock(h);
  OUTPUT:
    RETVAL

UV
add_many(self, items)
    SV *self
    SV *items
  PREINIT:
    EXTRACT(self);
    AV *av;
    IV  top;
    UV  changed = 0;
  CODE:
    if (!SvROK(items) || SvTYPE(SvRV(items)) != SVt_PVAV)
        croak("Data::MinHash::Shared->add_many: expected an array reference");
    av = (AV *)SvRV(items);
    top = av_len(av);                     /* last index, -1 if empty */
    {
        STRLEN cnt = (top >= 0) ? (STRLEN)(top + 1) : 0, i;
        const char **ps = NULL; STRLEN *ls = NULL;
        if (cnt) {                                       /* resolve all bytes BEFORE locking */
            Newx(ps, cnt, const char *); SAVEFREEPV(ps); /* freed on return OR unwind */
            Newx(ls, cnt, STRLEN);       SAVEFREEPV(ls);
            for (i = 0; i < cnt; i++) {                  /* a croak here holds NO lock; SAVEFREEPV cleans up */
                SV **el = av_fetch(av, (SSize_t)i, 0);
                if (el && *el) ps[i] = SvPVbyte(*el, ls[i]);
                else { ps[i] = ""; ls[i] = 0; }
            }
        }
        mnh_rwlock_wrlock(h);                             /* locked region: NO croak-capable calls */
        for (i = 0; i < cnt; i++) changed += (UV)mnh_add_locked(h, ps[i], ls[i]);
        __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);  /* a call always counts, even an empty batch */
        mnh_rwlock_wrunlock(h);
    }
    RETVAL = changed;
  OUTPUT:
    RETVAL

double
similarity(self, other)
    SV *self
    SV *other
  PREINIT:
    EXTRACT(self);
  CODE:
    if (!sv_isobject(other) || !sv_derived_from(other, "Data::MinHash::Shared"))
        croak("Data::MinHash::Shared->similarity: expected a Data::MinHash::Shared object");
    MnhHandle *o = INT2PTR(MnhHandle*, SvIV(SvRV(other)));
    if (!o) croak("Attempted to use a destroyed Data::MinHash::Shared object");

    /* k is immutable after creation -- compare lock-free, croak BEFORE allocating
     * so a mismatch holds no lock and leaks no buffer. */
    uint64_t ok = o->hdr->k;
    if (ok != h->hdr->k)
        croak("Data::MinHash::Shared->similarity: register-count mismatch (k=%llu vs k=%llu)",
              (unsigned long long)h->hdr->k, (unsigned long long)ok);

    /* Snapshot the other's registers under its read lock into a temp buffer, then
     * release before taking self's read lock.  Copying to a temp avoids holding
     * two locks at once (deadlock-free regardless of acquisition order). */
    uint64_t k = ok;
    uint64_t o_max = mnh_reg_max(o);      /* Layer B: never read past o's real mapping */
    if (k > o_max) k = o_max;
    uint64_t *tmp;
    Newx(tmp, (size_t)(k ? k : 1), uint64_t);
    SAVEFREEPV(tmp);                      /* freed on normal return OR croak unwind */
    mnh_rwlock_rdlock(o);
    memcpy(tmp, mnh_registers(o), (size_t)k * sizeof(uint64_t));
    mnh_rwlock_rdunlock(o);

    uint64_t agree, kk = h->hdr->k;
    mnh_rwlock_rdlock(h);
    agree = mnh_agree_locked(h, tmp, k);
    mnh_rwlock_rdunlock(h);
    RETVAL = kk ? (double)agree / (double)kk : 0.0;
  OUTPUT:
    RETVAL

void
merge(self, other)
    SV *self
    SV *other
  PREINIT:
    EXTRACT(self);
  CODE:
    if (!sv_isobject(other) || !sv_derived_from(other, "Data::MinHash::Shared"))
        croak("Data::MinHash::Shared->merge: expected a Data::MinHash::Shared object");
    MnhHandle *o = INT2PTR(MnhHandle*, SvIV(SvRV(other)));
    if (!o) croak("Attempted to use a destroyed Data::MinHash::Shared object");

    /* k is immutable after creation -- compare lock-free, croak BEFORE allocating
     * so a mismatch holds no lock and leaks no buffer. */
    uint64_t ok = o->hdr->k;
    if (ok != h->hdr->k)
        croak("Data::MinHash::Shared->merge: register-count mismatch (k=%llu vs k=%llu)",
              (unsigned long long)h->hdr->k, (unsigned long long)ok);

    /* Snapshot the other's registers under its read lock into a temp buffer, then
     * release before taking self's write lock.  Copying to a temp avoids holding
     * two locks at once (deadlock-free regardless of acquisition order between
     * two processes merging each other). */
    uint64_t k = ok;
    uint64_t o_max = mnh_reg_max(o);      /* Layer B: never read past o's real mapping */
    if (k > o_max) k = o_max;
    uint64_t *tmp;
    Newx(tmp, (size_t)(k ? k : 1), uint64_t);
    SAVEFREEPV(tmp);                      /* freed on normal return OR croak unwind */
    mnh_rwlock_rdlock(o);
    memcpy(tmp, mnh_registers(o), (size_t)k * sizeof(uint64_t));
    mnh_rwlock_rdunlock(o);

    mnh_rwlock_wrlock(h);
    mnh_merge_locked(h, tmp, k);
    __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);
    mnh_rwlock_wrunlock(h);

void
registers(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  PPCODE:
    {
        uint64_t k = h->hdr->k;
        uint64_t kmax = mnh_reg_max(h);   /* Layer B: never read past the real mapping */
        if (k > kmax) k = kmax;
        uint64_t *snap = NULL;
        if (k) { Newx(snap, (size_t)k, uint64_t); SAVEFREEPV(snap); }  /* alloc BEFORE the lock */
        mnh_rwlock_rdlock(h);
        if (k) memcpy(snap, mnh_registers(h), (size_t)k * sizeof(uint64_t));
        mnh_rwlock_rdunlock(h);
        EXTEND(SP, (SSize_t)k);
        for (uint64_t j = 0; j < k; j++)
            PUSHs(sv_2mortal(newSVuv((UV)snap[j])));    /* Perl alloc AFTER unlock */
    }

UV
filled(self)
    SV *self
  PREINIT:
    EXTRACT(self);
    UV n;
  CODE:
    mnh_rwlock_rdlock(h);
    n = (UV)mnh_filled_locked(h);
    mnh_rwlock_rdunlock(h);
    RETVAL = n;
  OUTPUT:
    RETVAL

void
clear(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    mnh_rwlock_wrlock(h);
    mnh_clear_locked(h);
    __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);
    mnh_rwlock_wrunlock(h);

UV
size(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = (UV)h->hdr->k;
  OUTPUT:
    RETVAL

SV *
stats(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    {
        uint64_t k, filled, ops;
        /* Snapshot under the lock; do all (croak-capable) Perl allocation after
           releasing it -- so an OOM in newHV/newSVuv can never strand the lock. */
        mnh_rwlock_rdlock(h);
        filled = mnh_filled_locked(h);
        k      = h->hdr->k;
        ops    = h->hdr->stat_ops;
        mnh_rwlock_rdunlock(h);

        HV *hv = newHV();
        hv_stores(hv, "size",        newSVuv((UV)k));
        hv_stores(hv, "filled",      newSVuv((UV)filled));
        hv_stores(hv, "ops",         newSVuv((UV)ops));
        hv_stores(hv, "mmap_size",   newSVuv((UV)h->mmap_size));
        RETVAL = newRV_noinc((SV *)hv);
    }
  OUTPUT:
    RETVAL

SV *
path(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = h->path ? newSVpv(h->path, 0) : &PL_sv_undef;
  OUTPUT:
    RETVAL

int
memfd(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = h->backing_fd;
  OUTPUT:
    RETVAL

void
sync(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    if (mnh_msync(h) != 0) croak("sync: %s", strerror(errno));

void
unlink(self, ...)
    SV *self
  CODE:
    if (sv_isobject(self) && sv_derived_from(self, "Data::MinHash::Shared")) {
        MnhHandle *h = INT2PTR(MnhHandle*, SvIV(SvRV(self)));
        if (h && h->path) unlink(h->path);
    } else if (items >= 2 && (SvGETMAGIC(ST(1)), SvOK(ST(1)))) {
        unlink(SvPV_nolen(ST(1)));
    }
