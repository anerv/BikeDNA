#!/usr/bin/python

# Export all notebooks to HTML without code cells

# One optional parameter possible, to choose mode:
# 1: Only generate 1a and 1b
# 2: Only generate 2a and 2b
# 3: Generate 1a+1b and 2a+2b and 3a
# 4: Generate 1a+1b and 2a+2b and 3a+3b (default)
# Example: python export_notebooks2html.py 3

import argparse
parser = argparse.ArgumentParser(description='Export executed notebooks to HTML.', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('mode', metavar='mode', type=int, nargs='?',
                    help='The export mode. \n 1: Only generate 1a and 1b \n 2: Only generate 2a and 2b \n 3: Generate 1a+1b and 2a+2b and 3a \n 4: Generate 1a+1b and 2a+2b and 3a+3b (default)', default=4)
parser.add_argument('--forpdf', dest='forpdf', action='store_true',
                    help='Export extra htmls needed for pdf generation, \n and do not export interactive cells (default: no)')
args = parser.parse_args()
mode = args.mode
forpdf = args.forpdf

from traitlets.config import Config
import nbformat as nbf
from nbconvert.preprocessors import TagRemovePreprocessor, ExecutePreprocessor
from nbconvert.exporters import HTMLExporter
from nbconvert.exporters.templateexporter import default_filters
import sys, os
import subprocess

os.chdir("scripts/settings/")
exec(open("yaml_variables.py").read())
os.chdir("../../")
ipath = "scripts/"
if forpdf:
    opath = "exports/"+study_area+"/pdf/" # Export temporary htmls to the pdf folder
else:
    opath = "exports/"+study_area+"/html/"

# Configure
c = Config()
c.TagRemovePreprocessor.remove_cell_tags = ("noex",)
if forpdf: c.TagRemovePreprocessor.remove_cell_tags += ("interactive",)
c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)
c.TagRemovePreprocessor.enabled = True
c.TemplateExporter.exclude_input_prompt = True
c.TemplateExporter.exclude_input = True
c.TemplateExporter.exclude_output_prompt = True
c.HTMLExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]

def postprocess_html(h):
    """ Post-process html file h to fix text-align and source paths
    """
    subprocess.run(["sh", "templates/postprocess_html.sh", h])

def export_to_html(notebook_file, html_file):
    """ Export nb to html with exporter
    """

    def custom_clean_html(element):
        """ Turn clean_html into a noop, to fix svg export bug.
        Inspired by: https://github.com/jupyter/nbconvert/issues/1894#issuecomment-1334355109
        """
        return element.decode() if isinstance(element, bytes) else str(element)

    default_filters["clean_html"] = custom_clean_html
    exporter = HTMLExporter(config=c)
    exporter.register_preprocessor(TagRemovePreprocessor(config=c),True)

    print("Exporting "+notebook_file+" to "+html_file+"..")
    output, _ = exporter.from_filename(notebook_file)
    open(html_file, mode="w", encoding="utf-8").write(output)
    postprocess_html(html_file)

# Convert

# Preamble
if forpdf:
    print("Exporting preamble..")
    export_to_html("templates/preamble.ipynb", opath+"preamble.html")

# OSM htmls
if mode == 1 or mode == 3 or mode == 4:
    export_to_html(ipath+"OSM/1a_initialize_osm.ipynb", opath+"1a.html")
    export_to_html(ipath+"OSM/1b_intrinsic_analysis_osm.ipynb", opath+"1b.html")

# REFERENCE htmls
if mode == 2 or mode == 3 or mode == 4:
    export_to_html(ipath+"REFERENCE/2a_initialize_reference.ipynb", opath+"2a.html")
    export_to_html(ipath+"REFERENCE/2b_intrinsic_analysis_reference.ipynb", opath+"2b.html")

# COMPARE htmls
if mode == 3 or mode == 4:
    export_to_html(ipath+"COMPARE/3a_extrinsic_analysis_metrics.ipynb", opath+"3a.html")
if mode == 4:
    export_to_html(ipath+"COMPARE/3b_extrinsic_analysis_feature_matching.ipynb", opath+"3b.html")

# Appendix
if forpdf:
    print("Setting up appendix..")
    with open("templates/appendix_a.ipynb") as f:
        nb = nbf.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=1000, kernel_name='bikedna')
    ep.preprocess(nb, {'metadata': {'path': "templates/"}})
    with open(opath+'appendix_a.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    export_to_html(opath+'appendix_a.ipynb', opath+"appendix_a.html")
    os.remove(opath+'appendix_a.ipynb')

print("Done!")