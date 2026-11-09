# Third-party notices

The s2apler native extension (`s2apler._s2apler_rust`) includes code copied and adapted from
S2AND's native extension (`s2and._s2and_rust`): the sparse counter and its Jaccard overlap
(`src/counter.rs`), the string similarity functions (`src/text.rs`), the thread-pool cache
(`src/rayon_pool.rs`), the build script (`build.rs`), and the LightGBM row keys and zero
window of its booster (`src/distinct_rows.rs`). Its upstream project is
<https://github.com/allenai/S2AND>. S2AND is distributed under the MIT License:

```text
MIT License

Copyright (c) 2021 Allen Institute for Artificial Intelligence

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

The s2apler wheels for x86_64 Linux (glibc) and Windows include mimalloc,
through the `mimalloc` and `libmimalloc-sys` Rust crates, as their memory
allocator. Its upstream project is <https://github.com/microsoft/mimalloc>.
mimalloc is distributed under the MIT License:

```text
MIT License

Copyright (c) 2018-2025 Microsoft Corporation, Daan Leijen

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

The vendored cld2-sys 1.0.2 source makes detection debug state thread-local.
See vendor/cld2-sys/S2AND_PATCHES.md for the patch and supported detection settings.

The s2and musllinux wheels bundle the GCC C++ runtime, `libstdc++` and `libgcc_s`,
in `s2and.libs/`, because CLD2 is C++ and musllinux does not provide that runtime.
These libraries are part of GCC <https://gcc.gnu.org/> and are distributed under the
GNU General Public License version 3 with the GCC Runtime Library Exception, which
permits this redistribution.
