import language_tool_python

if __name__ == "__main__":
    tool = language_tool_python.LanguageTool('de-DE')

    tool_easy = language_tool_python.LanguageTool('de-DE-x-simple-language')

    text = """Im Zuge der Waffenruhe im Gazastreifen können einige Kranke und Verletzte das Krisengebiet verlassen. Der Grenzübergang Rafah wurde erstmals wieder geöffnet. Tausende Lkw warten darauf, Hilfsgüter in das Küstengebiet zu bringen.Seit rund zwei Wochen herrscht zwischen Israel und der Terrororganisation Hamas eine Waffenruhe. Mehrere Geiseln der Islamisten wurden bereits freigelassen - zuletzt drei Männer, die zurück nach Israel konnten. Jetzt der nächste Fortschritt: Erstmals seit fast neun Monaten ist der Grenzübergang Rafah zwischen Ägypten und dem Süden des Gazastreifens wieder geöffnet worden."""

    matches = tool_easy.check(text)

    print(matches)