from voice.language import detect_language, normalize, strip_wake_name


def test_english():
    assert detect_language("Open the browser and check my website").language == "en"
    assert detect_language("show me the files").language == "en"


def test_roman_urdu_and_mixed():
    assert detect_language("mujhe is business ki website bana kar do").language == "roman_ur"
    assert detect_language("Browser mein website check karo aur jo issues hain fix karo").language in ("mixed", "roman_ur")
    assert detect_language("Pehle plan banao, phir kaam shuru karo").language in ("mixed", "roman_ur")


def test_urdu_script():
    assert detect_language("ویب سائٹ بناؤ").language == "ur"


def test_normalize_spelling_variants_only_internally():
    original = "Mjhe files dikhaao kro"
    assert normalize(original) == "mujhe files dikhao karo"
    assert original == "Mjhe files dikhaao kro"


def test_english_words_not_rewritten():
    assert normalize("show me the files or folders") == "show me the files or folders"


def test_wake_name():
    assert strip_wake_name("Abbad, open notepad") == "open notepad"
    assert strip_wake_name("Hey Abbas open notepad") == "open notepad"
    assert strip_wake_name("open notepad") == "open notepad"
    # Observed faster-whisper mishearings
    assert strip_wake_name("A bass. Stop.") == "Stop."
    assert strip_wake_name("A bad, which windows are open?") == "which windows are open?"
