#ifndef LLIST_H
#define LLIST_H

struct llist;
typedef struct llist *LinkedList;

/* Creates a new linked list and returns a pointer to it */
struct llist *llist_new(void);

/* Return the element at the specified index */
void *llist_get(struct llist *, int);
/* Return and remove last element */
/* Return NULL if empty */
void *llist_pop(struct llist *);
/* Return last element of list */
/* Return NULL if empty */
void *llist_peek(struct llist *);

/* Adds a new element at the end of the list and returns that element */
/* Element must be dynamically allocated (malloc()) or NULL */
void *llist_add(struct llist *, void *);
void *llist_push(struct llist *, void *);

/* Adds a new element at the specified index and returns that element */
/* If index < 0, insert element at beginning, if index > count, insert at the end */
/* Element must be dynamically allocated (malloc()) or NULL */
void *llist_addat(struct llist *, void *, int);

/* Set the element at the specified index */
/* Return previous element (does not deallocate) */
void *llist_set(struct llist *, void *, int);

/* Removes the item at the specified index (shifting the index of the following items) and returns it */
/* Does not free() element */
void *llist_remove(struct llist *, int);

/* Returns the number of elements in the list */
int llist_count(struct llist *);

/* Removes all items in the linked list and free()s all elements (unless element is NULL) */
/* Returns list */
/* llist_clear_f provides an option to release the elements with the specified function. */
struct llist *llist_clear(struct llist *);
struct llist *llist_clear_f(struct llist *, void (*)(void *));

/* Returns pointer to dynamically allocated array of all elements */
void **llist_toarray(struct llist *);

/* Creates a new linked list from and array and returns a pointer to it */
/* Elements must be dynamically allocated or NULL */
struct llist *llist_fromarray(void **, int);

/* Removes all items in the linked list and free()s all elements (unless element is NULL) */
/* Destroys linked list. */
/* Every llist_new() or llist_fromarray() call must have a matching llist_destroy() call. */
/* llist_destroy_f provides an option to release the elements using the specified function. */
void llist_destroy(struct llist *);
void llist_destroy_f(struct llist *, void (*)(void *));

#endif
