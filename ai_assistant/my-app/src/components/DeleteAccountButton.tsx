"use client"

export default function DeleteAccountButton() {
  async function deleteAccount() {
    if (!confirm("This permanently deletes your account. Continue?")) return
    const res = await fetch("/api/account/delete", { method: "POST" })
    if (res.ok) window.location.href = "/login"
    else alert("Could not delete account.")
  }

  return (
    <button
      onClick={deleteAccount}
      className="mt-6 border border-red-400 bg-red-50 text-red-700 px-3 py-2 rounded-md"
    >
      Delete my account
    </button>
  )
}
