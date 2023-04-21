import { Configuration, OpenAIApi } from 'openai';

const configuration = new Configuration({
  organization: process.env.OPENAI_ORG,
  apiKey: process.env.OPENAI_SECRET_KEY,
});

const openai = new OpenAIApi(configuration);

async function main() {
  console.log(`The original function being checked:

\`\`\`ts
export const sortModules = (modules: ModuleInterface[], existingModules: string[]) => {
  const moduleList = [...modules];
  const sortedModuleNames: { [key: string]: boolean } = {};
  const sortedModules = [];
  // Put all of the existing modules into the sortedModuleNames hash so they can be used for the
  // checks
  existingModules.forEach((m: string) => (sortedModuleNames[m] = true));
  do {
    const m = moduleList.shift();
    if (!m) break;
    if ((m.dependencies.length ?? 0) === 0 || m.dependencies.every(dep => sortedModuleNames[dep])) {
      sortedModuleNames[\`\${m.name}@\${m.version}\`] = true;
      sortedModules.push(m);
    } else {
      moduleList.push(m);
    }
  } while (moduleList.length > 0);
  return sortedModules;
};
\`\`\`
`);

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
export const sortModules = (modules: ModuleInterface[], existingModules: string[]) => {
  const moduleList = [...modules];
  const sortedModuleNames: { [key: string]: boolean } = {};
  const sortedModules = [];
  // Put all of the existing modules into the sortedModuleNames hash so they can be used for the
  // checks
  existingModules.forEach((m: string) => (sortedModuleNames[m] = true));
  do {
    const m = moduleList.shift();
    if (!m) break;
    if ((m.dependencies.length ?? 0) === 0 || m.dependencies.every(dep => sortedModuleNames[dep])) {
      sortedModuleNames[\`\${m.name}@\${m.version}\`] = true;
      sortedModules.push(m);
    } else {
      moduleList.push(m);
    }
  } while (moduleList.length > 0);
  return sortedModules;
};
\`\`\`
`
    }],
    model: 'gpt-3.5-turbo',
  });
  console.log(res.data.choices[0].message.content);

  const res2 = await openai.createChatCompletion({
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
export const sortModules = (modules: ModuleInterface[], existingModules: string[]) => {
  const moduleList = [...modules];
  const sortedModuleNames: { [key: string]: boolean } = {};
  const sortedModules = [];
  // Put all of the existing modules into the sortedModuleNames hash so they can be used for the
  // checks
  existingModules.forEach((m: string) => (sortedModuleNames[m] = true));
  do {
    const m = moduleList.shift();
    if (!m) break;
    if ((m.dependencies.length ?? 0) === 0 || m.dependencies.every(dep => sortedModuleNames[dep])) {
      sortedModuleNames[\`\${m.name}@\${m.version}\`] = true;
      sortedModules.push(m);
    } else {
      moduleList.push(m);
    }
  } while (moduleList.length > 0);
  return sortedModules;
};
\`\`\`
`
    }],
    model: 'gpt-3.5-turbo',
  });
  console.log(res2.data.choices[0].message.content);

  const res3 = await openai.createChatCompletion({
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
#1: ${res.data.choices[0].message.content}

#2: ${res2.data.choices[0].message.content}
`
    }],
    model: 'gpt-3.5-turbo',
  });
  console.log(res3.data.choices[0].message.content);
}

main();