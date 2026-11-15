# Deploy SupportGen online (VS Code + GitHub + Render, free tier)

## A. Send the project to GitHub from VS Code
1. File -> Open Folder -> choose this SupportGen-v2 folder.
2. Source Control icon (Ctrl+Shift+G) -> Initialize Repository.
3. Check the list: .env must NOT be there (it is ignored). Never upload your API key.
4. Type a message, click Commit (answer Yes to "stage all changes").
5. Click Publish Branch -> sign in to GitHub -> choose "private repository".

## B. Run it on Render
1. https://render.com -> sign up with GitHub -> New -> Blueprint -> pick your repo.
2. Paste GEMINI_API_KEY when asked (leave empty for offline demo mode) -> Apply.
3. Environment tab -> reveal the generated ADMIN_TOKEN (needed for Tickets/Upload/Delete buttons).
4. Open the https://<name>.onrender.com link when it says Live.

## C. Update later
Edit -> save -> Source Control -> Commit -> Sync Changes. Render redeploys automatically.

Free-tier facts: sleeps after ~15 min idle (first visit takes ~1 min), tickets and uploads are lost on restart,
one instance only. Your Gemini key pays for every chat, so share the link carefully.
Check https://render.com/pricing for current limits.
