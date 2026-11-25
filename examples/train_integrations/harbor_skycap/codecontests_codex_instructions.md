# Working on this task

Write `/app/solution.py` with `apply_patch`. A new file uses `*** Add File:`, and each of its lines is a `+`
followed directly by the line's text, with no space after the `+`. A space there becomes part of the line, so
every line would be indented and Python would raise IndentationError:

```
*** Begin Patch
*** Add File: /app/solution.py
+n = int(input())
+print(n * 2)
*** End Patch
```

`*** Update File:` is only for files that already exist, and its hunks have no `@@ -a,b +c,d @@` line numbers.

Before you finish, run `python3 /app/solution.py` on the example input from the problem and compare its output
with the expected output.

Always act through tool calls. Code that you write in a message is never saved.
