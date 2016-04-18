#!/bin/sh
find . -name "*.hpp" -o -name "*.h" -o -name "*.cpp" -o -name "*.cu" -o -name "*.c" -o -name "*.cc" > cscope.files
cscope -bkq -i cscope.files
ctags -R
