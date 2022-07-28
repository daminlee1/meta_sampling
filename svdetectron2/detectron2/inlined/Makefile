all:
	gcc -c -fPIC -I/usr/include/python2.7 launcher.c
	gcc -L/usr/lib/x86_64-linux-gnu -o launcher launcher.o -lpython2.7
	#gcc -static -I/usr/include/python2.7 -L/usr/lib/x86_64-linux-gnu -o launcher launcher.c -lpython2.7 -ldl -lm -lutil -lz -pthread
	#gcc -shared -o launcher.so launcher.o -lpython2.7 -Wl,-z,origin
	rm launcher.o
