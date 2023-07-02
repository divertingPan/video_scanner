from contextlib import suppress
with suppress(ModuleNotFoundError):
    import pyi_splash

    pyi_splash.update_text('loaded...')
    pyi_splash.close()


# pyinstaller command: pyinstaller -F -w --splash=<path-to-image> --icon=<path-to-icon> <your-script>.py
