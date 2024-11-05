@echo off
setlocal

rem Loop over each line in files.txt and format using Black with specified options
for /f "delims=" %%f in (cr_local/files.txt) do (
    echo Formatting %%f with Black
    black --line-length=100 --skip-string-normalization --preview -- %%f
)

echo Done!
