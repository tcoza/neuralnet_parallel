#include <stdlib.h>
#include <assert.h>
#include "../_ret1free2.c"
#include "llist.h"

struct _e_holder;

struct llist
{
	int count;
	struct _e_holder *first;

	// For optimization
	int curr_i;
	struct _e_holder *curr;
};

struct _e_holder
{
	struct _e_holder *next;
	void *element;
};

static struct _e_holder *_findholder(struct llist *list, int index);

void *llist_peek(struct llist *list) { return llist_get(list, list->count - 1); }

void *llist_pop(struct llist *list) { return llist_remove(list, list->count - 1); }

void *llist_push(struct llist *list, void *element) { return llist_add(list, element); }

void llist_destroy(struct llist *list) { llist_destroy_f(list, free); }

void llist_destroy_f(struct llist *list, void (*free_element)(void *)) { free(llist_clear_f(list, free_element)); }

struct llist *llist_fromarray(void **array, int size)
{
	struct llist *newlist = llist_new();
	for (int i = 0; i < size; i++)
		llist_add(newlist, array[i]);
	return newlist;
}

void **llist_toarray(struct llist *list)
{
	void **array = (void **)malloc(sizeof(void *) * list->count);
	int index = 0;
	for (struct _e_holder *curr = list->first; curr != NULL; curr = curr->next)
		array[index++] = curr->element;
	return array;
}

struct llist *llist_clear(struct llist *list) { return llist_clear_f(list, free); }

struct llist *llist_clear_f(struct llist *list, void (*free_element)(void *))
{
	for (struct _e_holder *curr = list->first; curr != NULL; curr = (struct _e_holder *)_ret1free2(curr->next, curr))
		free_element(curr->element);
	list->first = NULL;
	list->count = 0;
	list->curr = NULL;
	return list;
}

int llist_count(struct llist *list)
{
//	int realcount = 0;
//	for (struct _e_holder *curr = list->first; curr != NULL; curr = curr->next)
//		realcount++;
//	assert(list->count == realcount);

	return list->count;
}

void *llist_remove(struct llist *list, int index)
{
	if (index >= list->count || index < 0)
		return NULL;
	struct _e_holder *prev = _findholder(list, index-1);
	struct _e_holder *curr;

	if (prev == NULL)
	{
		curr = list->first;
		list->first = curr->next;
		list->curr_i--;
		if (list->curr_i < 0)
			list->curr = NULL;
	}
	else
	{
		curr = prev->next;
		prev->next = curr->next;
	}
	list->count--;

	void *ret = _ret1free2(curr->element, curr);
	return ret;
}

void *llist_addat(struct llist *list, void *element, int index)
{
	if (index < 0) index = 0;
	if (index >= list->count)
		return llist_add(list, element);
	struct _e_holder *prev = _findholder(list, index-1);
	struct _e_holder *newelement = (struct _e_holder *)malloc(sizeof(struct _e_holder));
	if (prev == NULL)
	{
		newelement->next = list->first;
		list->first = newelement;
		list->curr_i++;
	}
	else
	{
		newelement->next = prev->next;
		prev->next = newelement;
	}
	list->count++;

	return element;
}

void *llist_set(struct llist *list, void *element, int index)
{
	if (index < 0 || index >= list->count)
		return NULL;

	struct _e_holder *curr = _findholder(list, index);

	void *prevelement = curr->element;
	curr->element = element;
	return prevelement;
}

void *llist_add(struct llist *list, void *element)
{
	struct _e_holder *curr = (struct _e_holder *)malloc(sizeof(struct _e_holder));
	curr->element = element;
	curr->next = NULL;

	struct _e_holder *last = _findholder(list, list->count-1);

	if (last == NULL)
		list->first = curr;
	else
		last->next = curr;
	list->count++;

	return element;
}

void *llist_get(struct llist *list, int index)
{
	if (index < 0 || index >= list->count)
		return NULL;
	return _findholder(list, index)->element;
}

struct llist *llist_new()
{
	struct llist *newllist = (struct llist *)malloc(sizeof(struct llist));
	newllist->first = NULL;
	newllist->count = 0;
	newllist->curr = NULL;
	newllist->curr_i = -1;
	return newllist;
}

static struct _e_holder *_findholder(struct llist *list, int index)
{
	if (index < 0)
		return NULL;

	int i = 0;
	struct _e_holder *curr = list->first;
	if (list->curr != NULL && index >= list->curr_i)
	{
		i = list->curr_i;
		curr = list->curr;
	}

	for (; curr != NULL; curr = curr->next)
		if (i++ == index)
			break;

	if (curr != NULL)
	{
		list->curr_i = index;
		list->curr = curr;
	}

	return curr;
}
