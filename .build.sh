#!/bin/bash

# Bundles all of the dependencies for the `index.js` file into a `./build/main.js` file, then uses
# the new Single Executable Application functionality to build a binary named `git-marsha`
webpack ./index.js --target node --mode production -o ./build
node --experimental-sea-config sea-config.json
cp $(command -v node) git-marsha
npx postject git-marsha NODE_SEA_BLOB sea-prep.blob --sentinel-fuse NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2