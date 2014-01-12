CC	= g++
CFLAGS	= -Wall -std=c++11 -Wextra -pedantic -fopenmp
INCLUDES = -I/home/yuto/local/c++/include

.PHONY: check-syntax clean
check-syntax:
	$(CC) $(CFLAGS) $(INCLUDES) $(CHK_SOURCES) -fsyntax-only 
clean:
	rm -rf *.o *~ *.bak *.deps *.tgz a.out \#*\# \**\* *flymake*
.cpp.o: .cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -g

HHMM: HHMM.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o HHMM HHMM.cpp

TestHHMM: TestHHMM.cpp main.cpp HHMM.cpp TestHHMM.h HHMM.h Sequence.h
	$(CC) $(CFLAGS) $(INCLUDES) -o TestHHMM TestHHMM.cpp main.cpp -O3 -lcppunit
