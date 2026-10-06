int scanf(const char *restrict format, ...);
int printf(const char *restrict format, ...);
int puts(const char *s);

long input_int() {
    long x;
    int count = scanf("%ld", &x);
    if (count) {
      // printf("read: %ld\n", x);
      return x;
    }
    else {
      puts("input_int failed!\n\n");
	return 0;
    }

}

void print_int(long x) {
    printf("%ld\n", x);
}
