<script lang="ts">
  import '../app.css';
</script>

<nav>
  <a href="/" class="logo">Aragora</a>
  <div class="nav-links">
    <a href="/debates">Debates</a>
    <a href="/debates/new">New Debate</a>
  </div>
</nav>

<main class="container">
  <slot />
</main>

<style>
  nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    font-size: 1.5rem;
    font-weight: bold;
    color: var(--primary);
    text-decoration: none;
  }

  .nav-links {
    display: flex;
    gap: 1.5rem;
  }

  .nav-links a {
    color: var(--text-muted);
    text-decoration: none;
    transition: color 0.2s;
  }

  .nav-links a:hover {
    color: var(--text);
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }
</style>
