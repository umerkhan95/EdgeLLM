# Ollama API Gateway - Frontend

React-based frontend for the Ollama API Gateway with admin and user dashboards.

## Features

- 🔐 **API Key Authentication** - Secure sign-in with API keys
- 👥 **Role-Based Access** - Admin and user dashboards with different permissions
- 📊 **Analytics Dashboard** - Real-time usage statistics and charts
- 🎨 **Dark/Light Theme** - Toggle between dark and light modes
- 📈 **Data Visualization** - Chart.js powered charts for usage metrics
- 🔑 **API Key Management** - Create, view, and revoke API keys (admin only)

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **React Router** - Client-side routing
- **Tailwind CSS** - Styling
- **Chart.js** - Data visualization
- **Axios** - HTTP client
- **Lucide React** - Icons
- **date-fns** - Date formatting

## Setup

### Installation

```bash
cd frontend
npm install
```

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

### Development

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

The production build will be in the `dist` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── Navbar.jsx
│   │   ├── StatsCard.jsx
│   │   ├── Charts.jsx
│   │   └── APIKeyForm.jsx
│   ├── context/          # React context providers
│   │   ├── AuthContext.jsx
│   │   └── ThemeContext.jsx
│   ├── pages/            # Page components
│   │   ├── Home.jsx
│   │   ├── SignIn.jsx
│   │   ├── UserDashboard.jsx
│   │   └── AdminDashboard.jsx
│   ├── services/         # API service layer
│   │   └── api.js
│   ├── App.jsx           # Main app component
│   ├── main.jsx          # Entry point
│   └── index.css         # Global styles
├── public/               # Static assets
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## Usage

### Demo API Keys

For testing, you can use these demo API keys:

- **Admin**: `demo-admin-key-12345`
- **User**: `demo-user-key-67890`

### Admin Dashboard

Admins can:
- View all API keys
- Create new API keys
- Revoke existing API keys
- View system-wide statistics
- Monitor usage across all users

### User Dashboard

Users can:
- View their own usage statistics
- See requests over time
- Monitor rate limit usage
- View response time metrics
- Analyze requests by endpoint

## API Integration

The frontend communicates with the FastAPI backend at `http://localhost:8000`. All API calls include the `X-API-Key` header for authentication.

### Available Endpoints

- `GET /api/keys` - List API keys (requires authentication)
- `POST /api/keys` - Create new API key (admin only)
- `DELETE /api/keys/:id` - Revoke API key (admin only)
- `GET /api/stats` - Get basic statistics
- `GET /api/stats/detailed` - Get detailed analytics

## Customization

### Changing Theme Colors

Edit `tailwind.config.js` to customize the primary color:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        // Customize these values
        500: '#0ea5e9',
        600: '#0284c7',
        700: '#0369a1',
      }
    }
  }
}
```

### API Base URL

Change the API base URL in `src/services/api.js` or use the `VITE_API_URL` environment variable.

## License

MIT
