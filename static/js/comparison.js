function changeImages(event, cmpId, img1Src, img2Src) {
    // 防止按钮重复选中
    let buttons = document.querySelectorAll('.cmp-button');
    buttons.forEach(button => button.classList.remove('cmp-btn-checked'));
    event.target.classList.add('cmp-btn-checked');

    // 获取当前的比较容器
    const cmpContainer = document.getElementById(cmpId);
    
    // 更新图像
    const topImage = cmpContainer.querySelector('.top img');
    const bottomImage = cmpContainer.querySelector('.bottom img');
    
    topImage.src = img1Src;
    bottomImage.src = img2Src;
    
    // 重置滑动条到中间
    resetSlider(cmpContainer);
}

function resetSlider(cmpContainer) {
    const top = cmpContainer.querySelector(".top");
    const slider = cmpContainer.querySelector(".cmp-slider");
    const cmpWidth = cmpContainer.offsetWidth;
    
    // 将图片和滑动条重置为中间
    top.style.width = (cmpWidth) + "px";
    slider.style.left = (cmpWidth / 2) + "px";
}

// 初始化比较容器
window.onload = function() {
    initComparisons();
};

function initComparisons() {
    var cmpContainers = document.querySelectorAll(".cmp-container");
    
    cmpContainers.forEach(cmpContainer => {
        compareImages(cmpContainer);
    });

    function compareImages(cmpContainer) {
        var slider = cmpContainer.querySelector(".cmp-slider");
        var imgTop = cmpContainer.querySelector(".top");
        var clicked = 0, w = cmpContainer.offsetWidth;

        // 初始状态设置：左图和右图各显示 50%
        imgTop.style.clipPath = "inset(0 50% 0 0)";
        slider.style.left = (w / 2) + "px";

        slider.addEventListener("mousedown", slideReady);
        slider.addEventListener("touchstart", slideReady);
        window.addEventListener("mouseup", slideFinish);
        window.addEventListener("touchend", slideFinish);

        function slideReady(e) {
            e.preventDefault();
            clicked = 1;
            window.addEventListener("mousemove", slideMove);
            window.addEventListener("touchmove", slideMove);
        }

        function slideFinish() {
            clicked = 0;
        }

        function slideMove(e) {
            if (!clicked) return;
            var pos = getCursorPos(e);
            if (pos < 0) pos = 0;
            if (pos > w) pos = w;
            slide(pos);
        }

        function getCursorPos(e) {
            var rect = cmpContainer.getBoundingClientRect();
            return (e.pageX || e.touches[0].pageX) - rect.left;
        }

        function slide(x) {
            // 根据滑块位置调整左边和右边的图片显示区域
            //imgTop.style.clipPath = `inset(0 ${100 - (x / w) * 100}% 0 0)`; // 左图显示区域调整
            imgTop.style.clipPath = `inset(0 ${100 - (x / w) * 100}% 0 0)`;
            slider.style.left = x + "px"; // 移动滑块
        }
    }
}
