#include <stddef.h>
#include <stdatomic.h>
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t,size_t);
extern void *__libc_realloc(void*,size_t);
extern void __libc_free(void*);
static _Atomic unsigned long long calls, bytes;
void *malloc(size_t n){atomic_fetch_add(&calls,1);atomic_fetch_add(&bytes,n);return __libc_malloc(n);}
void *calloc(size_t m,size_t n){atomic_fetch_add(&calls,1);atomic_fetch_add(&bytes,m*n);return __libc_calloc(m,n);}
void *realloc(void *p,size_t n){atomic_fetch_add(&calls,1);atomic_fetch_add(&bytes,n);return __libc_realloc(p,n);}
void free(void *p){__libc_free(p);}
void ck_alloc_counter_reset(void){atomic_store(&calls,0);atomic_store(&bytes,0);}
unsigned long long ck_alloc_counter_calls(void){return atomic_load(&calls);}
unsigned long long ck_alloc_counter_bytes(void){return atomic_load(&bytes);}
