import Dashboard from "./pages/Dashboard";

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-neutral-200 bg-white">
        <div className="mx-auto max-w-5xl px-4 py-4">
          <h1 className="text-xl font-semibold">Cleaning Vision AI</h1>
          <p className="mt-1 text-sm text-neutral-600">
            Save a before photo, upload the after photo, review missing / new / damaged items.
          </p>
        </div>
      </header>
      <Dashboard />
    </div>
  );
}
