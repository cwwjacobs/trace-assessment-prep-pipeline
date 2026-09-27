# Renders examples/demo.gif from a real run of examples/demo.py.
# Re-render with:  vhs examples/demo.tape   (run from the repository root)
Output examples/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1280
Set Height 800
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

Hide
Type "export PS1='$ ' DEMO_PACE=1.6 && clear"
Enter
Show

Sleep 600ms
Type "python3 examples/demo.py"
Sleep 400ms
Enter
Sleep 12s
