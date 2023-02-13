#!/usr/bin/python

# Export all notebooks to PDF without code cells and interactive outputs

# One optional parameter possible, to choose mode:
# 1: Only generate 1a and 1b
# 2: Only generate 2a and 2b
# 3: Generate 1a+1b and 2a+2b and 3a
# 4: Generate 1a+1b and 2a+2b and 3a+3b (default)
# Example: python export_notebooks2pdf.py 3

pdfoptions = {
    "format": "A4",
    "display_header_footer": True,
    "margin": {"top": "1.1in", "bottom": "1.1in", "left": "0.6in", "right": "0.5in"},
    "prefer_css_page_size": True,
    "print_background": True,
}

section_names = {
    "1a": "1a. Initialize OSM data",
    "1b": "1b. Intrinsic OSM analysis",
    "2a": "2a. Initialize reference data",
    "2b": "2b. Intrinsic reference analysis",
    "3a": "3a. Extrinsic analysis",
    "3b": "3b. Feature matching",
    "appendix_a": "Appendix A: config.yml",
    "preamble": "Preamble",
}

import argparse

parser = argparse.ArgumentParser(
    description="Export executed notebooks to PDF.",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "mode",
    metavar="mode",
    type=int,
    nargs="?",
    help="The export mode. \n 1: Only generate 1a and 1b \n 2: Only generate 2a and 2b \n 3: Generate 1a+1b and 2a+2b and 3a \n 4: Generate 1a+1b and 2a+2b and 3a+3b (default)",
    default=4,
)
args = parser.parse_args()
mode = args.mode

import sys, os

os.chdir("scripts/settings/")
exec(open("yaml_variables.py").read())
os.chdir("../../")
iopath = "exports/" + study_area + "/pdf/"

print("==== STEP 1: CREATING TEMPORARY HTML FILES ====")

import subprocess

# Run export_notebooks2html with --forpdf option
subprocess.run(["python", "export_notebooks2html.py", str(mode), "--forpdf"])

print("==== STEP 2: CONVERTING TEMPORARY HTML FILES TO PDF ====")

# Set templates
subprocess.run(["sh", "templates/settemplates_pdf.sh"])
with open(iopath + "header_template.html") as f:
    pdfoptions["header_template"] = f.read()
with open(iopath + "footer_template.html") as f:
    pdfoptions["footer_template"] = f.read()


def update_header(sec):
    """
    Adjusts header template with current section sec
    """
    subprocess.run(["sh", "templates/settemplates_pdf.sh"])
    with open(iopath + "header_template.html", "r") as f:
        filedata = f.read()
        filedata = filedata.replace("[section]", section_names[sec])
    with open(iopath + "header_template.html", "w") as f:
        f.write(filedata)
    with open(iopath + "header_template.html") as f:
        pdfoptions["header_template"] = f.read()


def fix_css(fpathin, fpathout, inplace=True):
    """
    Removes the @page css tag from an html file at fpathin
    and adjusts font sizes. Saves as new html file at fpathout.
    """

    with open(fpathin, "r") as fin, open(fpathout, "w") as fout:
        lines = fin.readlines()
        wmode = 0
        for line in lines:
            if "@page {" in line:
                wmode = 3
            if wmode == 0:
                fout.write(line)
            elif wmode > 0:
                wmode -= 1
    subprocess.run(
        [
            "sed",
            "-i",
            "",
            "-e",
            "s/--jp-content-line-height: 1.6;/--jp-content-line-height: 1.5;/g",
            fpathout,
        ]
    )
    subprocess.run(
        [
            "sed",
            "-i",
            "",
            "-e",
            "s/--jp-content-font-size0: 0.83333em;/--jp-content-font-size0: 0.75em;/g",
            fpathout,
        ]
    )
    subprocess.run(
        [
            "sed",
            "-i",
            "",
            "-e",
            "s/--jp-content-font-size1: 14px;/--jp-content-font-size1: 12.6px;/g",
            fpathout,
        ]
    )
    subprocess.run(
        [
            "sed",
            "-i",
            "",
            "-e",
            "s/--jp-content-font-size2: 1.2em;/--jp-content-font-size2: 1.08em;/g",
            fpathout,
        ]
    )
    subprocess.run(
        [
            "sed",
            "-i",
            "",
            "-e",
            "s/--jp-content-font-size3: 1.44em;/--jp-content-font-size3: 1.296em;/g",
            fpathout,
        ]
    )
    subprocess.run(
        [
            "sed",
            "-i",
            "",
            "-e",
            "s/--jp-content-font-size5: 2.0736em;/--jp-content-font-size5: 2.5em;/g",
            fpathout,
        ]
    )
    subprocess.run(
        [
            "sed",
            "-i",
            "",
            "-e",
            "s/--jp-code-font-size: 13px/--jp-code-font-size: 11px;/g",
            fpathout,
        ]
    )
    if inplace:
        os.rename(fpathout, fpathin)


def convert_html2pdf(n, p):
    print("Converting " + n + ".html to pdf..")
    update_header(n)
    fix_css(iopath + n + ".html", iopath + n + "temp.html")
    p.goto("file://" + os.path.abspath(iopath + n + ".html"))
    p.wait_for_timeout(1000)
    p.pdf(path=iopath + n + ".pdf", **pdfoptions)


from playwright.sync_api import sync_playwright


def run(playwright):
    chromium = playwright.chromium  # or "firefox" or "webkit".
    browser = chromium.launch()
    page = browser.new_page()
    page.emulate_media(media="screen")

    # Titlepage
    print("Converting titlepage.html to pdf..")
    page.goto("file://" + os.path.abspath(iopath + "titlepage.html"))
    page.pdf(path=iopath + "titlepage.pdf", format=pdfoptions["format"])

    # Preamble
    convert_html2pdf("preamble", page)

    # OSM htmls
    if mode == 1 or mode == 3 or mode == 4:
        convert_html2pdf("1a", page)
        convert_html2pdf("1b", page)

    # REFERENCE htmls
    if mode == 2 or mode == 3 or mode == 4:
        convert_html2pdf("2a", page)
        convert_html2pdf("2b", page)

    # COMPARE htmls
    if mode == 3 or mode == 4:
        convert_html2pdf("3a", page)
    if mode == 4:
        convert_html2pdf("3b", page)

    # Appendix
    convert_html2pdf("appendix_a", page)

    browser.close()


with sync_playwright() as playwright:
    run(playwright)

print("==== STEP 3: FINALIZE ====")

# Stitch together
print("Stitching together single pdfs..")
args = [
    "gs",
    "-q",
    "-dNOPAUSE",
    "-dBATCH",
    "-dPDFSETTINGS=/prepress",
    "-sDEVICE=pdfwrite",
    "-sOutputFile=" + iopath + "report.pdf",
    iopath + "titlepage.pdf",
]
args.append(iopath + "preamble.pdf")
if mode == 1 or mode == 3 or mode == 4:
    args.append(iopath + "1a.pdf")
    args.append(iopath + "1b.pdf")
if mode >= 2:
    args.append(iopath + "2a.pdf")
    args.append(iopath + "2b.pdf")
if mode >= 3:
    args.append(iopath + "3a.pdf")
if mode >= 4:
    args.append(iopath + "3b.pdf")
args.append(iopath + "appendix_a.pdf")

subprocess.run(args)
args[4] = "-dPDFSETTINGS=/ebook"
args[6] = "-sOutputFile=" + iopath + "report_lowres.pdf"
subprocess.run(args)

# Remove temporary files
print("Removing temporary files..")
tempfiles = [
    "titlepage.html",
    "titlepage.pdf",
    "titleimage.svg",
    "footer_template.html",
    "header_template.html",
    "preamble.html",
    "preambletemp.html",
    "preamble.pdf",
    "1a.html",
    "1atemp.html",
    "1b.html",
    "1btemp.html",
    "2a.html",
    "2atemp.html",
    "2b.html",
    "2btemp.html",
    "3a.html",
    "3atemp.html",
    "3b.html",
    "3btemp.html",
    "appendix_a.html",
    "appendix_atemp.html",
    "appendix_a.pdf",
    "logo.svg"
]
for t in tempfiles:
    try:
        os.remove(iopath + t)
    except OSError:
        pass

print("Done! Created report.pdf and report_lowres.pdf in folder " + iopath)
