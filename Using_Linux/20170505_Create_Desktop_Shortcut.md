Yesterday I installed Qt Creator for Qt 5.8 on my Linux laptop. Everything was all right but I couldn't to figure out how to conveniently start this IDE, needing to go to the *bin* directory to click the executable file. I've tried to start it from commond line, but the IDE will crash when openning a dialog window. So I decided to create a desktop shortcut to bypass.

To manually create a desktop shortcut, we can create a .desktop file and place it in either /usr/share/applications or ~/.local/share/applications. A typical .desktop file looks like the following.

	[Desktop Entry]
	Encoding=UTF-8
	Version=1.0                                     # version of an app.
	Name[en_US]=yEd                                 # name of an app.
	GenericName=GUI Port Scanner                    # longer name of an app.
	Exec=java -jar /opt/yed-3.11.1/yed.jar          # command used to launch an app.
	Terminal=false                                  # whether an app requires to be run in a terminal.
	Icon[en_US]=/opt/yed-3.11.1/icons/yicon32.png   # location of icon file.
	Type=Application                                # type.
	Categories=Application;Network;Security;        # categories in which this app should be listed.
	Comment[en_US]=yEd Graph Editor                 # comment which appears as a tooltip.
