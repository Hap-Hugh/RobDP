//
// Created by Xuan Chen on 2025/9/3.
//

#ifndef SESSION_MEM_H
#define SESSION_MEM_H

extern MemoryContext session_mem_context(void);

extern void *session_palloc(Size bytes);

extern void *session_palloc0(Size bytes);

extern void *session_repalloc(void *ptr, Size newbytes);

extern void session_pfree(void *ptr);

extern void session_mem_reset(void);

extern void session_mem_delete(void);

#endif // SESSION_MEM_H
