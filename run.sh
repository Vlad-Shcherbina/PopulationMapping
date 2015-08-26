set -e -x

    #-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC \
clang++ \
    --std=c++0x -W -Wall -Wno-sign-compare \
    -O2 -pipe -mmmx -msse -msse2 -msse3 \
    -ggdb \
    -fsanitize=address,integer,undefined \
    -fno-sanitize-recover \
    main.cpp -o main

#java -jar tester/tester.jar -exec ./main -seed 4 -scale 3 -debug
java -classpath tester/patched_tester.jar PopulationMappingVis \
    -exec "./driver.sh" -seed 6 -scale 3
