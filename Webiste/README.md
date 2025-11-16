# Royal Enfield Model Identifier

AI-powered Royal Enfield motorcycle identification system with detailed bike information.

## Setup Instructions

### 1. Environment Setup

1. **Clone the repository** and navigate to the project directory
2. **Copy the environment template**:
   ```bash
   copy .env.example .env
   ```
3. **Get your Gemini API key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the API key

4. **Update the .env file**:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Security Features

✅ **API Key Protection**:
- API key stored in `.env` file (not in source code)
- `.env` file is gitignored (won't be uploaded to GitHub)
- API key is only used server-side (never exposed to client)

✅ **Direct REST API Usage**:
- Uses Google Gemini REST API directly
- No client-side API exposure
- Secure server-to-server communication

## Features

- **AI Model Identification**: TensorFlow-based Royal Enfield model recognition
- **Detailed Information**: Pricing, specifications, features via Gemini AI
- **Fallback System**: Local database when API is unavailable
- **Secure Configuration**: Environment-based API key management

## File Structure

```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Frontend template
├── static/
│   ├── style.css        # Styling
│   ├── script.js        # Frontend logic
│   └── images/          # Images and assets
├── .env                 # Environment variables (gitignored)
├── .env.example         # Environment template
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## API Security Notes

- The Gemini API key is never exposed to the client browser
- All API calls are made server-side only
- Environment variables are loaded securely using python-dotenv
- The `.env` file is automatically excluded from version control
