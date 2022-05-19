#!/bin/bash

# cleanup sources
rm -rf ./improveai.*.rst ./improveai.rst ./*.txt ./__pycache__

# cleanup html build
rm -rf ../.doctrees ../_sources ../_static ../_templates ../.buildinfo ../*.html ../*.js ../objects.inv

# cleanup post-package build stuff
rm -rf ../../.tox ../../dist ../../improveai.egg*

# rebuild docs rst files
sphinx-apidoc --separate -E -f -o . ../../improveai ../../improveai/old_files ../../improveai/experiments ../../improveai/tests ../../setup.py

FILES="*.rst"

for fn in $FILES
do
  # skip files which do not need a title
  if [ $fn == 'index.rst' ] ||
     [ $fn == 'modules.rst' ] ||
     [ $fn == 'improveai.rst' ] ||
     [ $fn == 'improveai.utils.rst' ] ||
     [ $fn == 'improveai.cythonized_feature_encoding.rst' ]; then
    continue
  fi

  # extract the python file name
  split_fn=($(echo "${fn}" | tr "." "\n"))
  fn_title=${split_fn[-2]}

  # set extensions
  extension="py"
  if [ "${fn_title}" == 'cythonized_feature_encoder' ] ||
     [ "${fn_title}" == 'cythonized_feature_encoding_utils' ]; then
    extension="pyx"
  fi

   # append title at the beginning of each file
   echo -e "${fn_title}.${extension} module\n============================================\n\n$(cat ${fn})" > "${fn}"
done

# rebuild html files
sphinx-build -v -E -b html ../_rst_sources ../.