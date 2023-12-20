1-if you have problems with refinitive, use yahoo finance for date. you can retrieve it in daily and monthly time.

2-i suggest you to save each file as name_m for monthly data or name_d for daily data.

3-Files are in csv, so i use this tutorial

https://www.youtube.com/watch?v=ecUEXE0OhqU

you have to follow it to make a union in a single(or multiple) file stocks and market values. it will make a union of each csv file that you want and it will separate it in sheets for each csv file.
i delete all coloum except the first and the "Close" coloums

it use macro in excel so you have to copy and past this visual basic code *****end of this file***** (copy without stars)

4-in the end have two file:
  #one with all stocks time series and each stock have a sheet in that file
  #one with market time series and interest rate(temporaly)

5-enjoy



*****
Sub CombineCsvFiles()
'updated by MelCompton
    Dim xFilesToOpen As Variant
    Dim I As Integer
    Dim xWb As Workbook
    Dim xTempWb As Workbook
    Dim xDelimiter As String
    Dim xScreen As Boolean
    On Error GoTo ErrHandler
    xScreen = Application.ScreenUpdating
    Application.ScreenUpdating = False
    xDelimiter = "|"
    xFilesToOpen = Application.GetOpenFilename("Text Files (*.csv), *.csv", , "MelCompton VBA for Excel", , True)
    If TypeName(xFilesToOpen) = "Boolean" Then
        MsgBox "No files were selected", , "MelCompton VBA for Excel"
        GoTo ExitHandler
    End If
    I = 1
    Set xTempWb = Workbooks.Open(xFilesToOpen(I))
    xTempWb.Sheets(1).Copy
    Set xWb = Application.ActiveWorkbook
    xTempWb.Close False
    Do While I < UBound(xFilesToOpen)
        I = I + 1
        Set xTempWb = Workbooks.Open(xFilesToOpen(I))
        xTempWb.Sheets(1).Move , xWb.Sheets(xWb.Sheets.Count)
    Loop
ExitHandler:
    Application.ScreenUpdating = xScreen
    Set xWb = Nothing
    Set xTempWb = Nothing
    Exit Sub
ErrHandler:
    MsgBox Err.Description, , "MelCompton VBA for Excel"
    Resume ExitHandler
End Sub

