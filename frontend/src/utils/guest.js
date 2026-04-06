const GUEST_ID_KEY = 'guest_id';

function generateGuestId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  return 'guest-' + Math.random().toString(16).slice(2) + Date.now().toString(16);
}

export function getGuestId() {
  if (typeof window === 'undefined') {
    return '';
  }

  let guestId = localStorage.getItem(GUEST_ID_KEY);
  if (!guestId) {
    guestId = generateGuestId();
    localStorage.setItem(GUEST_ID_KEY, guestId);
  }

  return guestId;
}

export function setGuestId(guestId) {
  if (typeof window === 'undefined' || !guestId) {
    return;
  }

  localStorage.setItem(GUEST_ID_KEY, guestId);
}
