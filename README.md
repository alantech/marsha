# MarSHA

An experiment in a code reviewing LLM bot.

## Design

Currently very early stage so subject to change, but at its core it would be a Map-Reduce engine.

For a given chunk of code, parallel ChatGPT (or similar LLM) calls would be executed, primed for different kinds of advice (Big-O optimization, improved type safety, improved security, comments matching the code, etc), and the results of this would be reduced into a singular recommendation by a special "combining" chat call executed log-parallel until all advice is merged into one result and then returned.

This could be pointed at a block of code by users in an editor, but the slow response time might make this annoying rather than helpful, so instead pointing at PRs and/or a cli pointed at specific files to chew on, where the latency would be more tolerable, seems like a more feasible early-stage version of this.