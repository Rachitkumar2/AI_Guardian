import ChangePasswordForm from '../../components/settings/ChangePasswordForm';

export default function SecuritySettings() {
  return (
    <div className="space-y-6 max-w-4xl">
      <div className="mb-4">
        <h2 className="text-[20px] font-bold text-white tracking-wide">Security & Access</h2>
        <p className="text-[14px] text-gray-400 mt-1">Manage your account authentication and password.</p>
      </div>

      <ChangePasswordForm />
    </div>
  );
}
