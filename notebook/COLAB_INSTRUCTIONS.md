# Publishing `toxbench_colab.ipynb` as a shared Google Colab notebook

Follow these steps in order. The whole process takes about 15 minutes, most of it waiting
for step 4 to finish running.

**What the grader needs at the end:** one link that an instructor or TA can open without a
Google account prompt, showing the notebook *with its results already displayed*, and still
runnable top to bottom if they choose to run it. Saving the notebook after a successful run
(step 6) is what puts the results in the link — skip it and the TA opens a notebook with no
output.

---

## 1. Upload the notebook to Colab

1. Open <https://colab.research.google.com> and sign in with the Google account you want to
   own this notebook.
2. Choose **File > Upload notebook**.
3. Select the **Upload** tab, click **Browse**, and pick
   `notebook/toxbench_colab.ipynb` from this project.
4. Colab opens the notebook and saves a copy in your Google Drive, under
   `My Drive / Colab Notebooks / toxbench_colab.ipynb`.

You do **not** need to upload any data, any ZIP file, or anything else. The notebook clones
the public repository <https://github.com/mzohaibcc/project606> itself when it runs.

## 2. Confirm the runtime

1. Choose **Runtime > Change runtime type**.
2. Set the hardware accelerator to **CPU**. The benchmark is CPU-only; a GPU runtime adds
   nothing and can be harder to obtain.
3. Click **Save**.

## 3. Check the repository is public

Open <https://github.com/mzohaibcc/project606> in a private or incognito browser window.
If the page loads without asking you to sign in, the notebook will be able to clone it. If
it shows a 404, the repository is private and **the notebook will fail for the TA** — make
it public in the repository's **Settings > General > Danger Zone > Change visibility**
before continuing.

## 4. Run the whole notebook

1. Choose **Runtime > Run all**.
2. Colab will warn that the notebook was not authored by Google. Click **Run anyway**.
3. **If a "RESTART SESSION" button or a restart prompt appears** after the dependency
   install in section 2.2 — or if section 2.3 stops with a message saying a package is
   loaded that is older than the one installed — do this:
   - choose **Runtime > Restart session**, then
   - choose **Runtime > Run all** again.

   This is normal and expected. Colab ships its own numpy and pandas, and replacing them
   under a running kernel only takes effect after a restart. Every cell is written to be
   idempotent: the second pass re-uses the clone and the already-installed packages and
   reaches the same point in seconds.
4. Wait for the run to finish. On a typical Colab CPU runtime expect roughly **8 to 15
   minutes** on a first pass: a couple of minutes to clone the repository, two to five
   minutes to install the pinned dependencies, and a few minutes for the live
   model-training section. These are estimates — Colab hands out different machines, so
   your run may be faster or slower.

## 5. Confirm the run succeeded

Scroll to the very bottom of the notebook. The final cell must print:

```
All protected directories untouched by this notebook: True
Figures a display cell could not find: 0
```

Also check, higher up:

* **section 2.3** ends with `All 9 pinned runtime packages match, and nothing stale is loaded.`
* **section 5.2** ends with `All 8 demonstration models completed with no failures.`
* **section 9.4** ends with `All committed figure files are present.`

If any cell shows a red error box, read its message — the notebook is written to say what
to do — fix it, and run again from **Runtime > Run all**. Do not share a notebook with a
failed cell in it.

## 6. Save the notebook with its results

Press **Ctrl+S** (or **Cmd+S** on a Mac), or choose **File > Save**.

This is the step that matters most for grading. Colab stores the cell outputs — every
table and every figure — inside the saved notebook, so the instructor and TA see the full
results the moment they open the link, without having to run anything. Wait until the
"Saving..." indicator next to the menu bar disappears.

> By default section 9.3 embeds six panels: the three precision-recall panels the project
> report names, and three reliability panels the cell selects from the committed data. To
> embed all 72 instead, set `SHOW_ALL_PANELS = True` in that cell and run it again before
> saving. The saved notebook then grows to roughly 35 MB and becomes slow to open, so the
> default of six representative panels is recommended. Either way, section 9.4 lists and
> verifies all 144 committed figure files.

## 7. Share the link

1. Click the **Share** button at the top right of the Colab window.
2. In the dialog, under **General access**, open the dropdown that currently reads
   *Restricted* and change it to **Anyone with the link**.
3. Next to it, make sure the role dropdown reads **Viewer**.
4. Click **Copy link**.
5. Click **Done**.

The copied link looks like
`https://colab.research.google.com/drive/1AbCdEfGhIjKlMnOpQrStUvWxYz...?usp=sharing`.

Viewer access is the right level: the instructor and TA can read the notebook and all its
saved results, and can run it themselves with **File > Save a copy in Drive**, but cannot
alter your submitted copy.

## 8. Verify the link works for someone who is not you

Do not skip this. A link that works while you are signed in can still be blocked for
everyone else.

1. Open a **private / incognito browser window** (Chrome: Ctrl+Shift+N, or Cmd+Shift+N on a
   Mac; Firefox and Safari have the same feature under File).
2. Paste the copied link and press Enter.
3. You should see the notebook with all its tables and figures. You should **not** see a
   sign-in wall, a "Request access" page, or a "You need permission" message.
4. Scroll to the bottom and confirm the saved results are visible there too — that proves
   step 6 worked.
5. If you are asked to sign in or request access, go back to step 7 and re-check that
   **General access** reads *Anyone with the link*, not *Restricted*.

## 9. Submit

1. Paste the link into the section of the project report titled **"Google Colab Notebook"**,
   as the course requirements specify.
2. Submit the report PDF and the same link through Canvas.

---

## Troubleshooting

| What you see | What to do |
| --- | --- |
| A restart prompt, or section 2.3 reports a `STALE IMPORT` | **Runtime > Restart session**, then **Runtime > Run all**. Expected on a first run. |
| Section 2.3 reports `WRONG VERSION` | Re-run the section 2.2 install cell and read pip's output for a resolution conflict, then re-run 2.3. Sections 6 to 11 read committed result files and do not depend on package versions; the cell explains how to continue past this deliberately if you need to. |
| `git clone failed` in section 2.1 | The Colab session has no network access, or the repository is not public. Check step 3, then re-run the cell. |
| Section 2.2 stops on the Python version | Colab's Python is outside the 3.12–3.14 window the pinned dependencies support. Colab's default usually is inside it; if this happens, note it and re-try later. |
| The run is very slow | A first pass clones about 130 MB and installs nine packages. Five to nine minutes is normal. |
| "Your session crashed after using all available RAM" | Choose **Runtime > Disconnect and delete runtime**, then start again from **Runtime > Run all**. The notebook's peak memory use is modest and this is usually a leftover session. |
| The saved notebook is too large to open quickly | Set `SHOW_ALL_PANELS = False` in section 9.3, re-run that cell, and save again. |

## What the notebook does and does not write

The notebook writes **only** to `outputs/notebook_demo/` inside its own cloned copy of the
repository, which lives in the Colab session and disappears when the session ends. It never
writes to `data/`, `partitions/`, `outputs/full/` or `outputs/smoke/`, it never modifies
your GitHub repository, and it never touches your Google Drive beyond the notebook file
itself. The final cell verifies this and fails loudly if anything protected was modified.
