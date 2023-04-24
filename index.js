import { execSync } from 'child_process';
import { readFileSync } from 'fs';
import { performance } from 'perf_hooks';

import ts from 'typescript';
import { Configuration, OpenAIApi } from 'openai';

const configuration = new Configuration({
  organization: process.env.OPENAI_ORG,
  apiKey: process.env.OPENAI_SECRET_KEY,
});

const openai = new OpenAIApi(configuration);
function checkNode(out, node) {
  if ([ts.SyntaxKind.FunctionDeclaration, ts.SyntaxKind.FunctionExpression, ts.SyntaxKind.MethodDeclaration].includes(node.kind)) {
    out.push(node);
  }
  ts.forEachChild(node, checkNode.bind(undefined, out));
}

function findRelevantNodes(node) {
  const out = [];
  checkNode(out, node);
  return out;
}

function getFilesToFunctions() {
  const t1 = performance.now();
  const fileAndLineNumbers = execSync(
    `git diff $(git remote show $(git remote show | head -n 1) | sed -n '/HEAD branch/s/.*HEAD branch: //p') --name-only | xargs -I '{}' echo '{}' | sed s"/\\(.*\\)/git blame \\1 | grep -nv '^\\\\^' | cut -f1 -d: | sed s\\/^\\/\\1:\\/g/g" | bash`,
    { encoding: 'utf8' },
  ).split('\n').map(r => r.split(':')).filter(r => /\.[jt]s$/.test(r[0]));
  const t2 = performance.now();
  console.log(`Time to get mutated lines: ${t2 - t1}`);
  const files = new Set(fileAndLineNumbers.map(r => r[0]));
  const filesToFunctions = new Map();
  for (const file of files) {
    const nodesToAnalyze = new Set();
    const lineNumbers = fileAndLineNumbers.filter(r => r[0] === file).map(r => r[1]);
    const sourceFile = ts.createSourceFile(file, readFileSync(file, 'utf8'));
    for (const statement of sourceFile.statements) {
      const relevantNodes = findRelevantNodes(statement);
      for (const node of relevantNodes) {
        for (const line of lineNumbers) {
          if (
            ts.getLineAndCharacterOfPosition(sourceFile, node.pos).line <= line &&
            ts.getLineAndCharacterOfPosition(sourceFile, node.end).line >= line
          ) {
            nodesToAnalyze.add(node);
          }
        }
      }
    }
    filesToFunctions.set(file, [...nodesToAnalyze].map(n => n.getText(sourceFile)));
    const t3 = performance.now();
    console.log(`Time to load, parse, and analyze all changed files: ${t3 - t2}`);
  }
  return filesToFunctions;
}

async function gptOptimize(func) {
  const res = await openai.createChatCompletion({
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

\`\`\`ts
${func}
\`\`\`
`
    }],
    model: 'gpt-3.5-turbo',
  });
  return res.data.choices[0].message.content;
}

async function gptType(func) {
  const res = await openai.createChatCompletion({
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

\`\`\`ts
${func}
\`\`\`
`
    }],
    model: 'gpt-3.5-turbo',
  });
  return res.data.choices[0].message.content;
}

async function gptReducer(rec1, rec2) {
  const res = await openai.createChatCompletion({
    messages: [{
      role: 'system',
      content: 'You are a senior software engineer helping review code. You are brief, answering with a simple \'No\' when nothing needs to be done and concise explanations otherwise.',
    }, {
      role: 'user',
      content: `I have received advice from different engineers on how to rewrite my code, #1 and #2 below. Can you combine them so I get both improvements?

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
`,
    }, {
      role: 'assistant',
      content: `This function's Big-O is \`O(n^2)\` due to the use of \`includes\`, which is \`O(n)\`, inside of a \`map\`, which is also \`O(n)\`. It also does not define the return type when there is only one return location and so its type is knowable. Finally, it uses the \`any\` type for two arrays of data that are passed into the same \`idGen\` function, implying that they both have the same underlying type for that function to be able to operate on both, which can be represented with type generics.

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
\`\`\`
`,
    }, {
      role: 'user',
      content: `I have received advice from different engineers on how to rewrite my code, #1 and #2 below. Can you combine them so I get both improvements?
#1: ${rec1}

#2: ${rec2}
`
    }],
    model: 'gpt-3.5-turbo',
  });
  return res.data.choices[0].message.content;
}

async function gptForFunc(func) {
  const baseRecs = await Promise.all([gptOptimize(func), gptType(func)]);
  return await gptReducer(...baseRecs);
}

async function gptForFile(funcs) {
  return await Promise.all(funcs.map(f => gptForFunc(f)));
}

async function main() {
  const filesToFunctions = getFilesToFunctions();

  const entries = [...filesToFunctions.entries()];
  const files = entries.map(e => e[0]);
  const funcs = entries.map(e => e[1]);
  const recs = await Promise.all(funcs.map(fs => gptForFile(fs)));
  for (let i = 0; i < files.length; i++) {
    console.log(`Recommendations for ${files[i]}`);
    const fileRecs = recs[i];
    for (let j = 0; j < fileRecs.length; j++) {
      console.log(fileRecs[j]);
      console.log('');
    }
  }
}

main();