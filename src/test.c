#include <stdio.h>


void foo(char w[]) {
    w++;
    printf("%c", *w);
    w++;
    printf("%c", *w);
}

void main() {
    char a[100];
    a[0] = 'a';
    a[1] = 'b';
    a[2] = 'c';
    a[4] = 'd';
    foo(a);
    return 0;
}
