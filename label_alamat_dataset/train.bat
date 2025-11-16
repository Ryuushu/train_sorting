@echo off

echo === Generate .box & .tr ===
for %%f in (train_ocr\*\*.png) do (
    echo Processing %%f
    tesseract %%f %%f nobatch box.train
)

echo === Extract unicharset ===
unicharset_extractor train_ocr\*\*.box

echo === Shape clustering ===
shapeclustering -F font_properties -U unicharset train_ocr\*\*.tr

echo === MFTraining ===
mftraining -F font_properties -U unicharset -O output.unicharset train_ocr\*\*.tr

echo === CNTraining ===
cntraining train_ocr\*\*.tr

echo === Rename output ===
rename normproto output.normproto
rename inttemp output.inttemp
rename pffmtable output.pffmtable
rename shapetable output.shapetable

echo === Combine Tessdata ===
combine_tessdata output.
