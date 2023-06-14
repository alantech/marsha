#!/bin/bash

# We need the *real* time, not the one built into the bash
# For now I'm being lazy and unrolling the loop since it's hardwired at 8 (it should be 30, but that
# will take forever)
/usr/bin/time --format='%e' -o time_test_1 ./dist/marsha $1
TEST1EXITCODE=$?
TEST1SUCCESS=`echo "$TEST1EXITCODE == 0" | bc`
TEST1TIME=`cat time_test_1 | sed 's/.* \([0-9]*\.[0-9]*\)/\1/'`
rm time_test_1
/usr/bin/time --format='%e' -o time_test_2 ./dist/marsha $1
TEST2EXITCODE=$?
TEST2SUCCESS=`echo "$TEST2EXITCODE == 0" | bc`
TEST2TIME=`cat time_test_2 | sed 's/.* \([0-9]*\.[0-9]*\)/\1/'`
rm time_test_2
/usr/bin/time --format='%e' -o time_test_3 ./dist/marsha $1
TEST3EXITCODE=$?
TEST3SUCCESS=`echo "$TEST3EXITCODE == 0" | bc`
TEST3TIME=`cat time_test_3 | sed 's/.* \([0-9]*\.[0-9]*\)/\1/'`
rm time_test_3
/usr/bin/time --format='%e' -o time_test_4 ./dist/marsha $1
TEST4EXITCODE=$?
TEST4SUCCESS=`echo "$TEST4EXITCODE == 0" | bc`
TEST4TIME=`cat time_test_4 | sed 's/.* \([0-9]*\.[0-9]*\)/\1/'`
rm time_test_4
/usr/bin/time --format='%e' -o time_test_5 ./dist/marsha $1
TEST5EXITCODE=$?
TEST5SUCCESS=`echo "$TEST5EXITCODE == 0" | bc`
TEST5TIME=`cat time_test_5 | sed 's/.* \([0-9]*\.[0-9]*\)/\1/'`
rm time_test_5
/usr/bin/time --format='%e' -o time_test_6 ./dist/marsha $1
TEST6EXITCODE=$?
TEST6SUCCESS=`echo "$TEST6EXITCODE == 0" | bc`
TEST6TIME=`cat time_test_6 | sed 's/.* \([0-9]*\.[0-9]*\)/\1/'`
rm time_test_6
/usr/bin/time --format='%e' -o time_test_7 ./distmarsha $1
TEST7EXITCODE=$?
TEST7SUCCESS=`echo "$TEST7EXITCODE == 0" | bc`
TEST7TIME=`cat time_test_7 | sed 's/.* \([0-9]*\.[0-9]*\)/\1/'`
rm time_test_7
/usr/bin/time --format='%e' -o time_test_8 ./dist/marsha $1
TEST8EXITCODE=$?
TEST8SUCCESS=`echo "$TEST8EXITCODE == 0" | bc`
TEST8TIME=`cat time_test_8 | sed 's/.* \([0-9]*\.[0-9]*\)/\1/'`
rm time_test_8

TOTALTIME=`echo "$TEST1TIME + $TEST2TIME + $TEST3TIME + $TEST4TIME + $TEST5TIME + $TEST6TIME + $TEST7TIME + $TEST8TIME" | bc`
AVGTIME=`echo "$TOTALTIME / 8.0" | bc`
SQ1=`echo "($TEST1TIME - $AVGTIME) ^ 2" | bc`
SQ2=`echo "($TEST2TIME - $AVGTIME) ^ 2" | bc`
SQ3=`echo "($TEST3TIME - $AVGTIME) ^ 2" | bc`
SQ4=`echo "($TEST4TIME - $AVGTIME) ^ 2" | bc`
SQ5=`echo "($TEST5TIME - $AVGTIME) ^ 2" | bc`
SQ6=`echo "($TEST6TIME - $AVGTIME) ^ 2" | bc`
SQ7=`echo "($TEST7TIME - $AVGTIME) ^ 2" | bc`
SQ8=`echo "($TEST8TIME - $AVGTIME) ^ 2" | bc`
STDDEV=`echo "sqrt(($SQ1 + $SQ2 + $SQ3 + $SQ4 + $SQ5 + $SQ6 + $SQ7 + $SQ8) / 8)" | bc`

echo Test results
echo `echo "$TEST1SUCCESS + $TEST2SUCCESS + $TEST3SUCCESS + $TEST4SUCCESS + $TEST5SUCCESS + $TEST6SUCCESS + $TEST7SUCCESS + $TEST8SUCCESS" | bc` / 8 runs successful
echo Runtime of $AVGTIME +/- $STDDEV