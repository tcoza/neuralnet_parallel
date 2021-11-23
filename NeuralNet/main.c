#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <io.h>
#include "../GetL/getl.h"
#include "../strmat.c"
#include "../ErrorF/errorf.h"
#include "../Random/random.h"

#define MAIN
#include "commands.h"
#include "neuralnetwork.h"

int _main(int, char**);
char **tokenize(char *str);
void print_net(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	const char *usage = "%s netfile1 [netfile2 [...]]";

	if (argc > 1 && strmat(argv[1], "--help", NULL))
	{
		printUsage(usage, argv[0]);
		return 0;
	}

	loadednets = llist_new();
	loadedfiles = llist_new();
	selectednet = 0;
	selectedfile = 0;

	srand(time(NULL));
	stdinIsTTY = _isatty(fileno(stdin));
	stdoutIsTTY = _isatty(fileno(stdout));
	stderrIsTTY = _isatty(fileno(stderr));

	char *load_command[] = { "load", NULL };
	for (int i = 1; i < argc; i++)
	{
		load_command[1] = argv[i];
		load(2, load_command);
	}
	if (llist_count(loadednets) > 0)
		selectednet = 1;

	const char *prompt = "\nneuralnet> ";
	int quit = 0;
	while (!feof(stdin) && !quit)
	{
		if (stdinIsTTY) fprintf(stderr, "%s", prompt);		// Print prompt
		else if (!stderrIsTTY) fputc('\n', stderr);			// Else: print an empty new line every command cycle. For process sync
		fflush(stderr);

		char *line = getl();
		char **tokens = tokenize(line);
		free(line);

		int tcount = 0;			// Get number of tokens
		while (tokens[tcount] != NULL)
			tcount++;

		if (tcount == 0)
			goto cont;

		switch (strmat(tokens[0],
				"output",			// Case 1
				"info",				// Case 2
				"load",				// Case 3
				"unload",			// Case 4
				"select",			// Case 5
				"train",			// Case 6
				"save",				// Case 7
				"print",			// Case 8
				"new",				// Case 9
				"rename",			// Case 10
				"clone",			// Case 11
				"get",				// Case 12
				"set",				// Case 13
				"crossover",		// Case 14
				"mutate",			// Case 15
				"help",				// Case 16
				"quit", NULL))		// Case 17
		{
		case 1: output(tcount, tokens); break;
		case 2: info(tcount, tokens); break;
		case 3: load(tcount, tokens); break;
		case 4: unload(tcount, tokens); break;
		case 5: select_index(tcount, tokens); break;
		case 6: train(tcount, tokens); break;
		case 7: save(tcount, tokens); break;
		case 8: print_net(tcount, tokens); break;
		case 9: new_net(tcount, tokens); break;
		case 10: rename_net(tcount, tokens); break;
		case 11: clone_net(tcount, tokens); break;
		case 12: get(tcount, tokens); break;
		case 13: set(tcount, tokens); break;
		case 14: crossover(tcount, tokens); break;
		case 15: mutate(tcount, tokens); break;
		case 16: help(); break;
		case 17: quit=1; break;
		default:
			errPut(COMMAND_NOT_FOUND, tokens[0]);
			break;
		}
	cont:
		tcount = 0;
		while (tokens[tcount] != NULL)
			free(tokens[tcount++]);
		free(tokens);

		fflush(stdout);
	}
	if (!stdinIsTTY && !stderrIsTTY)
		fputc('\n', stderr);

	llist_destroy_f(loadednets, (void (*)(void *))loadednet_free);
	llist_destroy_f(loadedfiles, (void (*)(void *))loadedfile_free);

	return 0;
}

void print_net(int argc, char *argv[])
{
	const char *usage = "%s";
	if (argc != 1)
		{ printUsage(usage, argv[0]); return; }

	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }

	NeuralNetwork *net = getLoadedNet(selectednet)->net;
	for (int L = 0; L < net->numberOfLayers; L++)
	{
		matrix_print(net->layers[L].weights);
		matrix_print(net->layers[L].biases);
	}
}

// Useful tools from here on...

#include <string.h>
#include <ctype.h>
#include <assert.h>

// Returns null-terminated dynamically allocated array of dynamically allocated strings.
char **tokenize(char *str)
{
	char **tokens = (char **)malloc(sizeof(char *) * ((strlen(str) + 1) / 2 + 1));		// Allocating enough memory for any eventuality

	int str_i = 0;
	int tks_i = 0;
	const char esc_char = '\\';
	assert(!isspace(esc_char));
	while (str[str_i] != '\0')		// While not reached end of string
	{
		while (isspace(str[str_i]))
			str_i++;		// Skip to next non-whitespace character
		if (str[str_i] == '\0')
			break;			// Reached end of string
		char *tk = (char *)malloc(sizeof(char) * (strlen(str) - str_i + 1));		// Allocating enough memory for any eventuality
		int tk_i = 0;
		int escape = 0;
		while (str[str_i] != '\0')
		{
			if (isspace(str[str_i]) && !escape)
				break;			// Found non-escaped whitespace character
			escape = !escape && (str[str_i] == esc_char);
			if (!escape) tk[tk_i++] = str[str_i];
			str_i++;
		}
		tk[tk_i] = '\0';		// Place terminator
		tk = (char *)realloc(tk, strlen(tk) + 1);

		tokens[tks_i++] = tk;
	}
	tokens[tks_i] = NULL;		// End array

	return (char **)realloc(tokens, sizeof(char *) * (tks_i + 1));		// Don't waste memory
}
