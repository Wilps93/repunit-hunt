/*==============================================================================
 * rh_arena.c — арена переиспользуемых mpz_t.
 * mpz_init вызывается РОВНО ОДИН РАЗ за жизнь потока; далее только
 * _mpz_realloc до нужного размера (тоже один раз). В горячем цикле
 * mpz_init/mpz_clear отсутствуют полностью.
 *============================================================================*/
#include "rh_prp.h"
#include <gmp.h>
#include <stdlib.h>
#include <string.h>

#define RH_SLOTS 16

struct rh_prp_arena {
    mpz_t    v[RH_SLOTS];
    uint64_t reserved_bits;
    char*    sbuf;
    size_t   scap;
    uint64_t last_base, last_k;   /* кэш построенного N */
    int      n_valid;
};

rh_prp_arena_t* rh_prp_arena_new(void) {
    rh_gmp_pool_install();                  /* ставим pool-аллокатор для потока */
    rh_prp_arena_t* a = (rh_prp_arena_t*)calloc(1,sizeof(*a));
    if(!a) return NULL;
    for(int i=0;i<RH_SLOTS;++i) mpz_init(a->v[i]);
    return a;
}

void rh_prp_arena_free(rh_prp_arena_t* a) {
    if(!a) return;
    for(int i=0;i<RH_SLOTS;++i) mpz_clear(a->v[i]);
    free(a->sbuf);
    free(a);
}

void rh_prp_arena_reserve(rh_prp_arena_t* a, uint64_t bits) {
    if(!a || bits <= a->reserved_bits) return;
    /* 2x запас: промежуточные произведения в powm занимают до 2*bits */
    uint64_t want = bits*2 + 256;
    mp_size_t limbs = (mp_size_t)((want + GMP_NUMB_BITS - 1)/GMP_NUMB_BITS);
    for(int i=0;i<RH_SLOTS;++i)
        if((mp_size_t)mpz_size(a->v[i]) < limbs) _mpz_realloc(a->v[i], limbs);
    a->reserved_bits = bits;
}

mpz_ptr rh_slot(rh_prp_arena_t* a, int i) { return a->v[i]; }

char* rh_sbuf(rh_prp_arena_t* a, size_t need) {
    if(a->scap < need){
        char* p=(char*)realloc(a->sbuf,need);
        if(!p) return NULL;
        a->sbuf=p; a->scap=need;
    }
    return a->sbuf;
}

/* Кэш: если (base,k) те же — N не пересчитываем (экономит ~15% при
   verify_factor сразу после PRP). */
int  rh_cache_hit(rh_prp_arena_t* a, uint64_t b, uint64_t k) {
    return a->n_valid && a->last_base==b && a->last_k==k;
}
void rh_cache_set(rh_prp_arena_t* a, uint64_t b, uint64_t k) {
    a->last_base=b; a->last_k=k; a->n_valid=1;
}
void rh_cache_clear(rh_prp_arena_t* a) { a->n_valid=0; }