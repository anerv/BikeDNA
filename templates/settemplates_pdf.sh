#!/bin/bash

# Prepare titlepage, preamble, header, and footer templates for pdf report

if [[ -z $1 ]];
then 
    mode=4 # if run without parameter, choose mode 4
else
    mode=$1
fi

area_name=$(grep "area_name:" config.yml | cut -d'"' -f 2)
area_name=${area_name##area_name: }
study_area=$(grep "study_area:" config.yml | cut -d'"' -f 2)
study_area=${study_area##study_area: }

# Footer and header
cp templates/footer_metatemplate.html exports/"$study_area"/pdf/footer_template.html
cp templates/header_metatemplate.html exports/"$study_area"/pdf/header_template.html
footerheader_template=$(cat templates/footerheader_template.css | tr -s '\n' ' ')
sed -i "" -e "s/\[area_name\]/${area_name}/g" exports/"$study_area"/pdf/header_template.html
sed -i "" -e "s/\[footerheader_template\]/${footerheader_template}/g" exports/"$study_area"/pdf/footer_template.html

# Titlepage
cp templates/titlepage_template.html exports/"$study_area"/pdf/titlepage.html
sed -i "" -e "s/\[area_name\]/${area_name}/g" exports/"$study_area"/pdf/titlepage.html
sed -i "" -e "s/\[timestamp\]/$(date "+%Y-%m-%d %H:%M:%S")/g" exports/"$study_area"/pdf/titlepage.html

# Preamble
sed -i "" -e "s/text-align: left/text-align: justify/g" exports/"$study_area"/pdf/preamble.html
sed -i "" -e "s/src='..\/..\/images\//src='..\/..\/..\/images\//g" exports/"$study_area"/pdf/preamble.html

if [ $mode == 1 ];
then
	cp results/OSM/"$study_area"/maps_static/titleimage.svg exports/"$study_area"/pdf/titleimage.svg
elif [ $mode == 2 ];
then
	cp results/REFERENCE/"$study_area"/maps_static/titleimage.svg exports/"$study_area"/pdf/titleimage.svg
else
	cp results/COMPARE/"$study_area"/maps_static/titleimage.svg exports/"$study_area"/pdf/titleimage.svg
fi


cp images/BikeDNA_logo.svg exports/"$study_area"/pdf/logo.svg
