#!/bin/bash

# Post-processing HTML file after nbconvert:
# Fixing text-justify and image paths

sed -i "" -e "s/text-align: left/text-align: justify/g" $1
sed -i "" -e "s/src='..\/..\/images\//src='..\/..\/..\/images\//g" $1
sed -i "" -e "s/src='..\/..\/results\//src='..\/..\/..\/results\//g" $1