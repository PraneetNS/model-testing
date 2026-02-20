import { useState } from 'react'
import Dashboard from './Dashboard'
import { Auth } from './Auth'

function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('ml_guard_token'))

  const handleAuth = (newToken: string) => {
    localStorage.setItem('ml_guard_token', newToken)
    setToken(newToken)
  }

  const handleLogout = () => {
    localStorage.removeItem('ml_guard_token')
    setToken(null)
  }

  if (!token) {
    return <Auth onAuthSuccess={handleAuth} />
  }

  return (
    <Dashboard token={token} onLogout={handleLogout} />
  )
}

export default App
