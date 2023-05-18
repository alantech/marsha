import { execSync } from 'child_process';
import { readFileSync } from 'fs';
import { performance } from 'perf_hooks';

import ts from 'typescript';
import { Configuration, OpenAIApi } from 'openai';
import { highlight } from 'cli-highlight';

const configuration = new Configuration({
  organization: process.env.OPENAI_ORG,
  apiKey: process.env.OPENAI_SECRET_KEY,
});

const openai = new OpenAIApi(configuration);
function checkFnNode(out, node) {
  if ([ts.SyntaxKind.FunctionDeclaration, ts.SyntaxKind.FunctionExpression, ts.SyntaxKind.ClassDeclaration].includes(node.kind)) {
    out.push(node);
  }
  ts.forEachChild(node, checkFnNode.bind(undefined, out));
}

function findFunctionNodes(node) {
  const out = [];
  checkFnNode(out, node);
  return out;
}

function checkCallNode(out, node) {
  if (node.kind === ts.SyntaxKind.CallExpression) {
    out.push(node);
  }
  ts.forEachChild(node, checkCallNode.bind(undefined, out));
}

function findCallNodes(node) {
  const out = [];
  checkCallNode(out, node);
  return out;
}

function getFileAndLineNumbers() {
  const t1 = performance.now();
  const changedLines = `
REMOTE=$(git remote show | head -n 1)
MAIN=$(git remote show $\{REMOTE\} | sed -n '/HEAD branch/s/.*HEAD branch: //p')
FILES=$(git diff $\{MAIN\} --name-only)
COMMITS="00000000
$(git log HEAD...$\{REMOTE\}/$\{MAIN\} --oneline --no-decorate | sed 's/ .*//g')"

OUT=""
IFS="
"
for FILE in $FILES
do
  for COMMIT in $COMMITS
  do
    git blame $FILE | grep -n "$\{COMMIT\}" | sed 's/ .*//g' | cut -f1 -d: | sed "s/^/$\{FILE\}:/g"
  done
done
  `;
  const fileAndLineNumbers = execSync(changedLines, { encoding: 'utf8', env: process.env })
    .split('\n').map(r => r.split(':')).filter(r => /\.[jt]sx?$/.test(r[0]));
  const t2 = performance.now();
  console.log(`Time to get mutated lines: ${t2 - t1}`);
  return fileAndLineNumbers;
}

function getFunctionsFromJavascriptishFile(sourceFile) {
  const functionNodes = new Set();
  for (const statement of sourceFile.statements) {
    const relevantNodes = findFunctionNodes(statement);
    for (const node of relevantNodes) {
      functionNodes.add(node);
    }
  }
  return functionNodes;
}

class Fn {
  name;
  original;
  deps;
  sourceFile;
  generated;
  declaration;
  toEval;

  constructor(name, original, deps, sourceFile) {
    this.name = name;
    this.original = original;
    this.deps = deps;
    this.sourceFile = sourceFile;
  }

  getOriginalText() {
    return this.original.getText(this.sourceFile);
  }
}

function analyzeJavascriptishFile(file, lineNumbers) {
  const fns = new Map();
  const sourceFile = ts.createSourceFile(file, readFileSync(file, 'utf8'));
  const fnsToAnalyze = new Set();
  const functionNodes = getFunctionsFromJavascriptishFile(sourceFile);
  for (const node of functionNodes) {
    const callNodesForFn = findCallNodes(node);
    const fnDeps = new Set();
    for (const callNode of callNodesForFn) {
      if (ts.SyntaxKind.Identifier === callNode.expression.kind) { // Only consider direct fn calls for now
        fnDeps.add(callNode.expression.escapedText);
      }
    }
    const fn = new Fn(node.name.escapedText, node, [...fnDeps], sourceFile);
    fns.set(fn.name, fn);
    for (const line of lineNumbers) {
      if (
        ts.getLineAndCharacterOfPosition(sourceFile, node.pos).line <= line &&
        ts.getLineAndCharacterOfPosition(sourceFile, node.end).line >= line
      ) {
        fn.toEval = true;
        fnsToAnalyze.add(fn);
      }
    }
  }
  for (const fn of fns.values()) {
    for (let i = 0; i < fn.deps.length; i++) {
      const dep = fn.deps[i];
      if (fns.has(dep)) {
        fn.deps[i] = fns.get(dep);
      } else {
        fn.deps.splice(i, 1);
        i--;
      }
    }
  }
  return fnsToAnalyze;
}

function getFilesToFunctions(fileAndLineNumbers) {
  const t1 = performance.now();
  const files = new Set(fileAndLineNumbers.map(r => r[0]));
  const filesToFunctions = new Map();
  for (const file of files) {
    const lineNumbers = fileAndLineNumbers.filter(r => r[0] === file).map(r => r[1]);
    if (/\.[tj]sx?$/.test(file)) {
      const fnsToAnalyze = analyzeJavascriptishFile(file, lineNumbers);
      filesToFunctions.set(file, fnsToAnalyze);
    }
  }
  const t2 = performance.now();
  console.log(`Time to load, parse, and analyze all changed files: ${t2 - t1}`);
  return filesToFunctions;
}

function functionParses(filename, filestring) {
  try {
    const sourceFile =  ts.createSourceFile(filename, filestring);
    return !sourceFile.parseDiagnostics.length;
  } catch (_) {
    return false;
  }
}

async function retryChatCompletion(query, maxTries=3) {
  const t1 = performance.now();
  do {
    try {
      const out = await openai.createChatCompletion(query);
      const t2 = performance.now();
      console.log(`Chat Query took ${t2 - t1}ms, started at ${t1}, ms/chars = ${(t2 - t1) / (out.data.usage?.total_tokens ?? 9001)}`);
      return out;
    } catch (e) {
      maxTries--;
      if (!maxTries) throw e;
      await new Promise(r => setTimeout(r, 3000 / maxTries));
    }
  } while(maxTries);
}

async function retryCompletion(query, maxTries=3) {
  const t1 = performance.now();
  do {
    try {
      const out = await openai.createCompletion(query);
      const t2 = performance.now();
      console.log(`Completion Query took ${t2 - t1}ms, started at ${t1}, ms/chars = ${(t2 - t1) / (out.data.usage?.total_tokens ?? 9001)}`);
      return out;
    } catch (e) {
      maxTries--;
      if (!maxTries) throw e;
      await new Promise(r => setTimeout(r, 3000 / maxTries));
    }
  } while(maxTries);
}

async function gptDeclaration(func, decLines, retries = 3) {
  const res = await retryChatCompletion({
    messages: [{
      role: 'system',
      content: 'You are a senior software engineer helping review code. You are brief, answering with a simple \'No\' when nothing needs to be done and concise explanations otherwise.',
    }, {
      role: 'user',
      content: `Can you give me a Typescript declare statement for this function?

\`\`\`ts
export function findDiff(dbEntities, cloudEntities, idGen, comparator) {
  const entitiesInDbOnly = [];
  const entitiesInAwsOnly = [];
  const dbEntityIds = dbEntities.map(idGen);
  const cloudEntityIds = cloudEntities.map(idGen);
  // Everything in cloud and not in db is a potential delete
  const cloudEntNotInDb = cloudEntities.filter(e => !dbEntityIds.includes(idGen(e)));
  cloudEntNotInDb.map(e => entitiesInAwsOnly.push(e));
  // Everything in db and not in cloud is a potential create
  const dbEntNotInCloud = dbEntities.filter(e => !cloudEntityIds.includes(idGen(e)));
  dbEntNotInCloud.map(e => entitiesInDbOnly.push(e));
  // Everything else needs a diff between them
  const remainingDbEntities = dbEntities.filter(e => cloudEntityIds.includes(idGen(e)));
  const entitiesChanged = [];
  remainingDbEntities.map(dbEnt => {
    const cloudEntToCompare = cloudEntities.find(e => idGen(e) === idGen(dbEnt));
    if (!comparator(dbEnt, cloudEntToCompare)) {
      entitiesChanged.push({
        db: dbEnt,
        cloud: cloudEntToCompare,
      });
    }
  });
  return {
    entitiesInDbOnly,
    entitiesInAwsOnly,
    entitiesChanged,
  };
}
\`\`\`
`,
    }, {
      role: 'assistant',
      content: `This function does not define the return type when there is only one return location and so its type is knowable. It also has two arrays of data that are passed into the same \`idGen\` function, implying that they both have the same underlying type for that function to be able to operate on both, which can be represented with type generics. We can write a Typescript declare statement like this:

\`\`\`ts
declare function findDiff<T>(
  dbEntities: T[],
  cloudEntities: T[],
  idGen: (e: T) => string,
  comparator: (a: T, b: T) => boolean,
): {
  entitiesInDbOnly: T[];
  entitiesInAwsOnly: T[];
  entitiesChanged: {
    db: T;
    cloud: T;
  }[];
};
\`\`\`
`
    }, {
      role: 'user',
      content: `Can you give me a Typescript declare statement for this function?
${decLines && `
I have these declare statements for some of the functions used:

${decLines}

And here is the function:

`}
\`\`\`ts
${func}
\`\`\`
`
    }],
    model: 'gpt-3.5-turbo',
  });
  const fn = res.data.choices[0].message.content.split('```').find(r => /^ts\n/.test(r))?.replace(/^ts\n/, '');
  if (!!fn && !functionParses('test', fn) && retries) {
    return gptDeclaration(func, retries--);
  }
  return res.data.choices[0].message.content.split('```').map(r => {
    if (/^ts\n/.test(r)) {
      return r.replace(/^ts\n/, '');
    }
    return '';
  }).join('');
  // return res.data.choices[0].message.content;
}

async function gptOptimize(func, decLines, retries = 3) {
  const res = await retryChatCompletion({
    messages: [{
      role: 'system',
      content: 'You are a senior software engineer helping review code. You are brief, answering with a simple \'No\' when nothing needs to be done and concise explanations otherwise.',
    }, {
      role: 'user',
      content: `Can this function's Big-O notation be improved?

\`\`\`ts
export function findDiff(
  dbEntities: any[],
  cloudEntities: any[],
  idGen: (e: any) => string,
  comparator: (a: any, b: any) => boolean,
) {
  const entitiesInDbOnly: any[] = [];
  const entitiesInAwsOnly: any[] = [];
  const dbEntityIds = dbEntities.map(idGen);
  const cloudEntityIds = cloudEntities.map(idGen);
  // Everything in cloud and not in db is a potential delete
  const cloudEntNotInDb = cloudEntities.filter(e => !dbEntityIds.includes(idGen(e)));
  cloudEntNotInDb.map(e => entitiesInAwsOnly.push(e));
  // Everything in db and not in cloud is a potential create
  const dbEntNotInCloud = dbEntities.filter(e => !cloudEntityIds.includes(idGen(e)));
  dbEntNotInCloud.map(e => entitiesInDbOnly.push(e));
  // Everything else needs a diff between them
  const remainingDbEntities = dbEntities.filter(e => cloudEntityIds.includes(idGen(e)));
  const entitiesChanged: any[] = [];
  remainingDbEntities.map(dbEnt => {
    const cloudEntToCompare = cloudEntities.find(e => idGen(e) === idGen(dbEnt));
    if (!comparator(dbEnt, cloudEntToCompare)) {
      entitiesChanged.push({
        db: dbEnt,
        cloud: cloudEntToCompare,
      });
    }
  });
  return {
    entitiesInDbOnly,
    entitiesInAwsOnly,
    entitiesChanged,
  };
}
\`\`\`
`,
    }, {
      role: 'assistant',
      content: `This function's Big-O is \`O(n^2)\` due to the use of \`includes\`, which is \`O(n)\`, inside of a \`map\`, which is also \`O(n)\`.

Using an object or \`Map\` can reduce this down to \`O(n)\`.

\`\`\`ts
export function findDiff(
  dbEntities: any[],
  cloudEntities: any[],
  idGen: (e: any) => string,
  comparator: (a: any, b: any) => boolean,
) {
  const dbEntitiesById = new Map(dbEntities.map(e => [idGen(e), e]));
  const cloudEntitiesById = new Map(cloudEntities.map(e => [idGen(e), e]));
  const remainingEntitiesById = new Map();
  const entitiesInDbOnly: any[] = [];
  const entitiesInAwsOnly: any[] = [];
  const entitiesChanged: any[] = [];
  for (const entityIdAndVal of dbEntitiesById) {
    if (cloudEntitiesById.has(entityIdAndVal[0])) {
      remainingEntitiesById.set(entityIdAndVal[0], [entityIdAndVal[1], cloudEntitiesById.get(entityIdAndVal[0])]);
    } else {
      entitiesInDbOnly.push(entityIdAndVal[1]);
    }
  }
  for (const entityIdAndVal of cloudEntitiesById) {
    if (!dbEntitiesById.has(entityIdAndVal[0])) {
      entitiesInAwsOnly.push(entityIdAndVal[1]);
    }
  }
  for (const remainingEntitiesRecord of remainingEntitiesById.values()) {
    const [dbEnt, cloudEnt] = remainingEntitiesRecord;
    if (!comparator(dbEnt, cloudEnt)) {
      entitiesChanged.push({
        db: dbEnt,
        cloud: cloudEnt,
      });
    }
  }
  return {
    entitiesInDbOnly,
    entitiesInAwsOnly,
    entitiesChanged,
  };
}
\`\`\`
`
    }, {
      role: 'user',
      content: `Can this function's Big-O notation be improved?
${decLines && `
I have these declare statements for some of the functions used:

\`\`\`ts
${decLines}
\`\`\`

And here is the function:

`}
\`\`\`ts
${func}
\`\`\`
`
    }],
    model: 'gpt-3.5-turbo',
  });
  const fn = res.data.choices[0].message.content.split('```').find(r => /^ts\n/.test(r))?.replace(/^ts\n/, '');
  if (!!fn && !functionParses('test', fn) && retries) {
    return gptOptimize(func, decLines, retries--);
  }
  return res.data.choices[0].message.content;
}

async function gptType(func, decLines, retries = 3) {
  const res = await retryChatCompletion({
    messages: [{
      role: 'system',
      content: 'You are a senior software engineer helping review code. You are brief, answering with a simple \'No\' when nothing needs to be done and concise explanations otherwise.',
    }, {
      role: 'user',
      content: `Can this function's type annotation be improved?

\`\`\`ts
export function findDiff(
  dbEntities: any[],
  cloudEntities: any[],
  idGen: (e: any) => string,
  comparator: (a: any, b: any) => boolean,
) {
  const entitiesInDbOnly: any[] = [];
  const entitiesInAwsOnly: any[] = [];
  const dbEntityIds = dbEntities.map(idGen);
  const cloudEntityIds = cloudEntities.map(idGen);
  // Everything in cloud and not in db is a potential delete
  const cloudEntNotInDb = cloudEntities.filter(e => !dbEntityIds.includes(idGen(e)));
  cloudEntNotInDb.map(e => entitiesInAwsOnly.push(e));
  // Everything in db and not in cloud is a potential create
  const dbEntNotInCloud = dbEntities.filter(e => !cloudEntityIds.includes(idGen(e)));
  dbEntNotInCloud.map(e => entitiesInDbOnly.push(e));
  // Everything else needs a diff between them
  const remainingDbEntities = dbEntities.filter(e => cloudEntityIds.includes(idGen(e)));
  const entitiesChanged: any[] = [];
  remainingDbEntities.map(dbEnt => {
    const cloudEntToCompare = cloudEntities.find(e => idGen(e) === idGen(dbEnt));
    if (!comparator(dbEnt, cloudEntToCompare)) {
      entitiesChanged.push({
        db: dbEnt,
        cloud: cloudEntToCompare,
      });
    }
  });
  return {
    entitiesInDbOnly,
    entitiesInAwsOnly,
    entitiesChanged,
  };
}
\`\`\`
`,
    }, {
      role: 'assistant',
      content: `This function does not define the return type when there is only one return location and so its type is knowable. It also uses the \`any\` type for two arrays of data that are passed into the same \`idGen\` function, implying that they both have the same underlying type for that function to be able to operate on both, which can be represented with type generics.

\`\`\`ts
export function findDiff<T>(
  dbEntities: T[],
  cloudEntities: T[],
  idGen: (e: T) => string,
  comparator: (a: T, b: T) => boolean,
): {
  entitiesInDbOnly: T[];
  entitiesInAwsOnly: T[];
  entitiesChanged: {
    db: T;
    cloud: T;
  }[];
} {
  const entitiesInDbOnly: T[] = [];
  const entitiesInAwsOnly: T[] = [];
  const dbEntityIds = dbEntities.map(idGen);
  const cloudEntityIds = cloudEntities.map(idGen);
  // Everything in cloud and not in db is a potential delete
  const cloudEntNotInDb = cloudEntities.filter(e => !dbEntityIds.includes(idGen(e)));
  cloudEntNotInDb.map(e => entitiesInAwsOnly.push(e));
  // Everything in db and not in cloud is a potential create
  const dbEntNotInCloud = dbEntities.filter(e => !cloudEntityIds.includes(idGen(e)));
  dbEntNotInCloud.map(e => entitiesInDbOnly.push(e));
  // Everything else needs a diff between them
  const remainingDbEntities = dbEntities.filter(e => cloudEntityIds.includes(idGen(e)));
  const entitiesChanged: { db: T; cloud: T; }[] = [];
  remainingDbEntities.map(dbEnt => {
    const cloudEntToCompare = cloudEntities.find(e => idGen(e) === idGen(dbEnt));
    if (!comparator(dbEnt, cloudEntToCompare)) {
      entitiesChanged.push({
        db: dbEnt,
        cloud: cloudEntToCompare,
      });
    }
  });
  return {
    entitiesInDbOnly,
    entitiesInAwsOnly,
    entitiesChanged,
  };
}
\`\`\`
`,
    }, {
      role: 'user',
      content: `Can this function's type annotation be improved?
${decLines && `
I have these declare statements for some of the functions used:

\`\`\`ts
${decLines}
\`\`\`

And here is the function:

`}
\`\`\`ts
${func}
\`\`\`
`
    }],
    model: 'gpt-3.5-turbo',
  });
  const fn = res.data.choices[0].message.content.split('```').find(r => /^ts\n/.test(r))?.replace(/^ts\n/, '');
  if (!!fn && !functionParses('test', fn) && retries) {
    return gptType(func, decLines, retries--);
  }
  return res.data.choices[0].message.content;
}

async function gptReducer(rec1, rec2, retries = 3) {
    const prompt = `
The following is a discussion between a senior software engineer (SENIOR) and a junior software engineer (JUNIOR).

JUNIOR: I have received advice from different engineers on how to rewrite my code, #1 and #2 below. Can you combine them so I get both improvements?

#1: This function does not define the return type when there is only one return location and so its type is knowable. It also uses the \`any\` type for two arrays of data that are passed into the same \`idGen\` function, implying that they both have the same underlying type for that function to be able to operate on both, which can be represented with type generics.

\`\`\`ts
export function findDiff<T>(
  dbEntities: T[],
  cloudEntities: T[],
  idGen: (e: T) => string,
  comparator: (a: T, b: T) => boolean,
): {
  entitiesInDbOnly: T[];
  entitiesInAwsOnly: T[];
  entitiesChanged: {
    db: T;
    cloud: T;
  }[];
} {
  const entitiesInDbOnly: T[] = [];
  const entitiesInAwsOnly: T[] = [];
  const dbEntityIds = dbEntities.map(idGen);
  const cloudEntityIds = cloudEntities.map(idGen);
  // Everything in cloud and not in db is a potential delete
  const cloudEntNotInDb = cloudEntities.filter(e => !dbEntityIds.includes(idGen(e)));
  cloudEntNotInDb.map(e => entitiesInAwsOnly.push(e));
  // Everything in db and not in cloud is a potential create
  const dbEntNotInCloud = dbEntities.filter(e => !cloudEntityIds.includes(idGen(e)));
  dbEntNotInCloud.map(e => entitiesInDbOnly.push(e));
  // Everything else needs a diff between them
  const remainingDbEntities = dbEntities.filter(e => cloudEntityIds.includes(idGen(e)));
  const entitiesChanged: T[] = [];
  remainingDbEntities.map(dbEnt => {
    const cloudEntToCompare = cloudEntities.find(e => idGen(e) === idGen(dbEnt));
    if (!comparator(dbEnt, cloudEntToCompare)) {
      entitiesChanged.push({
        db: dbEnt,
        cloud: cloudEntToCompare,
      });
    }
  });
  return {
    entitiesInDbOnly,
    entitiesInAwsOnly,
    entitiesChanged,
  };
}
\`\`\`

#2: This function's Big-O is \`O(n^2)\` due to the use of \`includes\`, which is \`O(n)\`, inside of a \`map\`, which is also \`O(n)\`.

Using an object or \`Map\` can reduce this down to \`O(n)\`.

\`\`\`ts
export function findDiff(
  dbEntities: any[],
  cloudEntities: any[],
  idGen: (e: any) => string,
  comparator: (a: any, b: any) => boolean,
) {
  const dbEntitiesById = new Map(dbEntities.map(e => [idGen(e), e]));
  const cloudEntitiesById = new Map(cloudEntities.map(e => [idGen(e), e]));
  const remainingEntitiesById = new Map();
  const entitiesInDbOnly: any[] = [];
  const entitiesInAwsOnly: any[] = [];
  const entitiesChanged: any[] = [];
  for (const entityIdAndVal of dbEntitiesById) {
    if (cloudEntitiesById.has(entityIdAndVal[0])) {
      remainingEntitiesById.set(entityIdAndVal[0], [entityIdAndVal[1], cloudEntitiesById.get(entityIdAndVal[0])]);
    } else {
      entitiesInDbOnly.push(entityIdAndVal[1]);
    }
  }
  for (const entityIdAndVal of cloudEntitiesById) {
    if (!dbEntitiesById.has(entityIdAndVal[0])) {
      entitiesInAwsOnly.push(entityIdAndVal[1]);
    }
  }
  for (const remainingEntitiesRecord of remainingEntitiesById.values()) {
    const [dbEnt, cloudEnt] = remainingEntitiesRecord;
    if (!comparator(dbEnt, cloudEnt)) {
      entitiesChanged.push({
        db: dbEnt,
        cloud: cloudEnt,
      });
    }
  }
  return {
    entitiesInDbOnly,
    entitiesInAwsOnly,
    entitiesChanged,
  };
}
\`\`\`

SENIOR: This function's Big-O is \`O(n^2)\` due to the use of \`includes\`, which is \`O(n)\`, inside of a \`map\`, which is also \`O(n)\`. It also does not define the return type when there is only one return location and so its type is knowable. Finally, it uses the \`any\` type for two arrays of data that are passed into the same \`idGen\` function, implying that they both have the same underlying type for that function to be able to operate on both, which can be represented with type generics.

\`\`\`ts
export function findDiff<T>(
  dbEntities: T[],
  cloudEntities: T[],
  idGen: (e: T) => string,
  comparator: (a: T, b: T) => boolean,
): {
  entitiesInDbOnly: T[];
  entitiesInAwsOnly: T[];
  entitiesChanged: {
    db: T;
    cloud: T;
  }[];
} {
  const dbEntitiesById: Map<string, T> = new Map(dbEntities.map(e => [idGen(e), e]));
  const cloudEntitiesById: Map<string, T> = new Map(cloudEntities.map(e => [idGen(e), e]));
  const remainingEntitiesById: Map<string, [T, T]> = new Map();
  const entitiesInDbOnly: T[] = [];
  const entitiesInAwsOnly: T[] = [];
  const entitiesChanged: { db: T; cloud: T; }[] = [];
  for (const entityIdAndVal of dbEntitiesById) {
    if (cloudEntitiesById.has(entityIdAndVal[0])) {
      remainingEntitiesById.set(entityIdAndVal[0], [entityIdAndVal[1], cloudEntitiesById.get(entityIdAndVal[0])]);
    } else {
      entitiesInDbOnly.push(entityIdAndVal[1]);
    }
  }
  for (const entityIdAndVal of cloudEntitiesById) {
    if (!dbEntitiesById.has(entityIdAndVal[0])) {
      entitiesInAwsOnly.push(entityIdAndVal[1]);
    }
  }
  for (const remainingEntitiesRecord of remainingEntitiesById.values()) {
    const [dbEnt, cloudEnt] = remainingEntitiesRecord;
    if (!comparator(dbEnt, cloudEnt)) {
      entitiesChanged.push({
        db: dbEnt,
        cloud: cloudEnt,
      });
    }
  }
  return {
    entitiesInDbOnly,
    entitiesInAwsOnly,
    entitiesChanged,
  };
}

JUNIOR: I have also received advice from different engineers on how to rewrite other code, #1 and #2 below. Can you combine them so I get both improvements?

#1: ${rec1}

#2: ${rec2}

SENIOR: `;
  const res = await retryCompletion({
    max_tokens: 4096 - Math.ceil(prompt.length / 3) - 100, // I don't know why some of these characters are counting as two bytes. I thought it was all UTF-8 8-bit chars? TODO: Find a more accurate token calculator
    prompt,
    model: 'text-davinci-003',
  });
  const fn = res.data.choices[0].text.split('```').find(r => /^ts\n/.test(r))?.replace(/^ts\n/, '');
  if (!!fn && !functionParses('test', fn) && retries) {
    return gptReducer(rec1, rec2, retries--);
  }
  return res.data.choices[0].text;
}

async function gptForFunc(func) {
  if (func.generated) { // Already processed this in a prior iteration
    return func.generated;
  }
  // Establish a perhaps not as good declaration to be used if there are cycles in the graph
  func.declaration = gptDeclaration(func.getOriginalText());
  // First, confirm all dependencies are fully analyzed, otherwise, recurse into the dependencies
  for (const dep of func.deps) {
    if (dep.declaration) continue;
    gptForFunc(dep);
  }
  const decLines = (await Promise.all([...func.deps].map(d => d.declaration))).join('\n');
  const original = func.getOriginalText();
  if (func.toEval) {
    const baseRecs = await Promise.all([gptOptimize(original, decLines), gptType(original, decLines)]);
    func.generated = gptReducer(...baseRecs);
    func.declaration = gptDeclaration(await func.generated);
  } else {
    func.generated = func.getOriginalText();
  }
  return func.generated;
}

async function gptForFile(funcs) {
  return await Promise.all(funcs.map(f => gptForFunc(f)));
}

async function main() {
  const fileAndLineNumbers = getFileAndLineNumbers();
  const filesToFunctions = getFilesToFunctions(fileAndLineNumbers);

  const entries = [...filesToFunctions.entries()];
  const files = entries.map(e => e[0]);
  const funcs = entries.map(e => [...e[1]]);
  const t1 = performance.now();
  const recs = await Promise.all(funcs.map(fs => gptForFile(fs)));
  const t2 = performance.now();
  for (let i = 0; i < files.length; i++) {
    console.log(`Recommendations for ${files[i]}`);
    const fileRecs = recs[i];
    for (let j = 0; j < fileRecs.length; j++) {
      // Only ts for now, so split that way
      console.log(fileRecs[j].split('```').map(r => {
        if (/^ts\n/.test(r)) {
          return highlight(r.replace(/^ts\n/, ''), { language: 'ts' });
        }
        return r;
      }).join(''));
      console.log('');
    }
  }
  console.log(`GPT time was ${(t2 - t1) / 1000}sec`);
}

main();