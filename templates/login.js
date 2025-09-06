function githubLogin() {
    window.location.href = '/auth/github/login';
}


// 添加此函数
function logoutGithub() {
    // 打开 GitHub 登出页面
    window.open('https://github.com/logout', '_blank');
    // 提示用户
    alert('请在新窗口中完成 GitHub 登出，然后关闭该窗口并返回此页面重新登录');
}

async function sendVerificationCode() {
    const email = document.getElementById('email').value;
    const btn = document.getElementById('sendCodeBtn');

    if (!email) {
        showMessage('请输入邮箱地址', 'error');
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '发送中 <span class="loading"></span>';

    try {
        const response = await fetch('/send-verification-code/', {
        method: 'POST',
        body: new URLSearchParams({email}),
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
    });

    const data = await response.json();

    if (response.ok) {
        showMessage('验证码已发送到您的邮箱', 'success');
        // 开始倒计时
        startCountdown(60);
    } else {
        showMessage(data.detail || '验证码发送失败', 'error');
    }
} catch (error) {
    showMessage('网络错误，请重试', 'error');
} finally {
    btn.disabled = false;
    btn.innerHTML = '获取验证码';
}
}

async function verifyCode() {
const email = document.getElementById('email').value;
const code = document.getElementById('code').value;

if (!email) {
    showMessage('请输入邮箱地址', 'error');
    return;
}

if (!code) {
    showMessage('请输入验证码', 'error');
    return;
}

try {
    const response = await fetch('/verify-code/', {
        method: 'POST',
        body: new URLSearchParams({email, code}),
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
    });

    const data = await response.json();

    if (response.ok) {
        localStorage.setItem('access_token', data.access_token);
        // 用后端返回的redirect_url跳转
        if (data.redirect_url) {
            window.location.href = data.redirect_url;
        } else {
            showMessage('登录成功，但未返回跳转地址', 'success');
        }
    } else {
        showMessage(data.detail || '验证码错误', 'error');
    }
} catch (error) {
    showMessage('网络错误，请重试', 'error');
}
}

function showMessage(message, type) {
const statusEl = document.getElementById('statusMessage');
statusEl.textContent = message;
statusEl.className = 'status-message ' + type;
}

function startCountdown(seconds) {
const btn = document.getElementById('sendCodeBtn');
btn.disabled = true;

let remaining = seconds;
btn.textContent = `${remaining}s后重新获取`;

const timer = setInterval(() => {
    remaining--;
    btn.textContent = `${remaining}s后重新获取`;

    if (remaining <= 0) {
        clearInterval(timer);
        btn.disabled = false;
        btn.textContent = '获取验证码';
    }
}, 1000);
}

// 按Enter键提交表单
document.getElementById('loginForm').addEventListener('keypress', function(e) {
if (e.key === 'Enter') {
    e.preventDefault();
    verifyCode();
}
});