import requests


def upgrade_bot_account(token):
    url = "https://lichess.org/api/bot/account/upgrade"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.post(url, headers=headers)

    if response.status_code == 200:
        print("Bot account upgraded successfully!")
    else:
        print(f"Error upgrading bot account: {response.status_code} - {response.text}")


# Replace '<yourTokenHere>' with your actual Lichess API token
your_token = "lip_Wb8PuphOUKejuROrhAfL"
upgrade_bot_account(your_token)