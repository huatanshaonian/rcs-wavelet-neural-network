@echo Y|cacls %* /t /e /c /g Everyone:f
 
DEL /F /A /Q \\?\%1
 
RD /S /Q \\?\%1