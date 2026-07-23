import re


def normalize_email(email):

    email = str(email).lower().strip()

    if "@" not in email:
        return email, ""

    username, domain = email.split("@", 1)

    username = re.sub(r"\+.*", "", username)

    username = re.sub(
        r"[._-]",
        "",
        username
    )

    return username, domain


def split_alpha_num(username):

    alpha = "".join(
        re.findall(r"[a-z]+", username)
    )

    nums = "".join(
        re.findall(r"\d+", username)
    )

    return alpha, nums


def normalize_matchpoint(text):
    if not text:
        return ""

    text = str(text).lower()
    text = text.replace(".com", "")

    return text.strip()