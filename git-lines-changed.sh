#!/bin/bash

# git diff $(git remote show $(git remote show | head -n 1) | sed -n '/HEAD branch/s/.*HEAD branch: //p') --name-only | sed s"/\\(.*\\)/git blame \\1 | grep -n '^$(git log HEAD...$(git remote show)/$(git remote show $(git remote show | head -n 1) | sed -n '/HEAD bran    ch/s/.*HEAD branch: //p') --oneline --no-decorate) | sed 's/ .*//g' | cut -f1 -d: | sed s\\/^\\/\\1:\\/g/g" | bash

REMOTE=$(git remote show | head -n 1)
MAIN=$(git remote show ${REMOTE} | sed -n '/HEAD branch/s/.*HEAD branch: //p')
FILES=$(git diff ${MAIN} --name-only)
COMMITS="00000000
$(git log HEAD...${REMOTE}/${MAIN} --oneline --no-decorate | sed 's/ .*//g')"

OUT=""
IFS=$'\n'
for FILE in $FILES
do
  for COMMIT in $COMMITS
  do
    git blame $FILE | grep -n "${COMMIT}" | sed 's/ .*//g' | cut -f1 -d: | sed "s/^/${FILE}:/g"
  done
done