# Third-Party Licenses

Aragora is MIT-licensed (see `LICENSE`). Components or patterns derived from
third-party projects are credited here.

## elves (aigorahub/elves)

- **Project:** https://github.com/aigorahub/elves
- **License:** MIT
- **Author:** John Ennis / aigorahub
- **Used in:** `.agents/skills/elves-aragora/`

The `elves-aragora` skill adapts the *scaffolding pattern* introduced by elves —
the three-document structure (Plan / Survival Guide / Execution Log) and the
crash/compaction-survival discipline for long unattended runs. No elves source
code is vendored; the aragora skill reimplements the pattern and binds every
validation gate to aragora's own governance (model-quorum evidence, decision
receipts, draft-PR-only, the Operating Contract). The upstream elves skill and
its templates are installed separately at `~/.claude/skills/elves/`.

The upstream project is distributed under the MIT License. A copy of the MIT
License text governing the adapted pattern is reproduced below for completeness.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
