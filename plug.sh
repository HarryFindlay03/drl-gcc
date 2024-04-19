#
# bash script to build and copy statetool plugin as a shared dynamic library object.
#
# Note: to be ran from root

echo "Building new stateplugin library"

cd "stateplugin"

make statetool

mv statetool.dylib ../statetool.dylib

echo "New stateplugin build - named statetool.dylib and found in root"


