//
// Created by Xuan Chen on 2025/9/3.
//

#include "postgres.h"
#include "utils/memutils.h"
#include "optimizer/session_mem.h"

static MemoryContext g_session_ctx = NULL;

static void
ensure_ctx(void) {
    if (g_session_ctx == NULL) {
        g_session_ctx = AllocSetContextCreate(
            TopMemoryContext,
            "MySessionMem",
            ALLOCSET_DEFAULT_SIZES
        );
    }
}

MemoryContext
session_mem_context(void) {
    elog(LOG, "session_mem_context");
    ensure_ctx();
    return g_session_ctx;
}

void *
session_palloc(Size bytes) {
    ensure_ctx();
    MemoryContext old = MemoryContextSwitchTo(g_session_ctx);
    void *p = palloc(bytes);
    MemoryContextSwitchTo(old);
    return p;
}

void *
session_palloc0(Size bytes) {
    ensure_ctx();
    MemoryContext old = MemoryContextSwitchTo(g_session_ctx);
    void *p = palloc0(bytes);
    MemoryContextSwitchTo(old);
    return p;
}

void *
session_repalloc(void *ptr, Size newbytes) {
    ensure_ctx();
    MemoryContext old = MemoryContextSwitchTo(g_session_ctx);
    void *p = repalloc(ptr, newbytes);
    MemoryContextSwitchTo(old);
    return p;
}

void
session_pfree(void *ptr) {
    if (ptr)
        pfree(ptr);
}

void
session_mem_reset(void) {
    if (g_session_ctx)
        MemoryContextReset(g_session_ctx);
}

void
session_mem_delete(void) {
    if (g_session_ctx) {
        MemoryContextDelete(g_session_ctx);
        g_session_ctx = NULL;
    }
}
