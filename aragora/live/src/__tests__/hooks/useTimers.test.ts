import { renderHook, act } from '@testing-library/react';
import {
  useTimeout,
  useTimeoutEffect,
  useInterval,
  useIntervalEffect,
  useDebounce,
  useThrottle,
} from '@/hooks/useTimers';

describe('useTimeout', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should execute callback after delay', () => {
    const callback = jest.fn();
    const { result } = renderHook(() => useTimeout());

    act(() => {
      result.current.setTimeout(callback, 1000);
    });

    expect(callback).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(1000);
    });

    expect(callback).toHaveBeenCalledTimes(1);
  });

  it('should allow clearing timeout before it fires', () => {
    const callback = jest.fn();
    const { result } = renderHook(() => useTimeout());

    let timeoutId: NodeJS.Timeout;
    act(() => {
      timeoutId = result.current.setTimeout(callback, 1000);
    });

    act(() => {
      result.current.clearTimeout(timeoutId);
    });

    act(() => {
      jest.advanceTimersByTime(2000);
    });

    expect(callback).not.toHaveBeenCalled();
  });

  it('should clean up all timeouts on unmount', () => {
    const callback1 = jest.fn();
    const callback2 = jest.fn();
    const { result, unmount } = renderHook(() => useTimeout());

    act(() => {
      result.current.setTimeout(callback1, 1000);
      result.current.setTimeout(callback2, 2000);
    });

    unmount();

    act(() => {
      jest.advanceTimersByTime(3000);
    });

    expect(callback1).not.toHaveBeenCalled();
    expect(callback2).not.toHaveBeenCalled();
  });

  it('should handle multiple concurrent timeouts', () => {
    const callback1 = jest.fn();
    const callback2 = jest.fn();
    const { result } = renderHook(() => useTimeout());

    act(() => {
      result.current.setTimeout(callback1, 500);
      result.current.setTimeout(callback2, 1500);
    });

    act(() => {
      jest.advanceTimersByTime(600);
    });

    expect(callback1).toHaveBeenCalledTimes(1);
    expect(callback2).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(1000);
    });

    expect(callback2).toHaveBeenCalledTimes(1);
  });
});

describe('useTimeoutEffect', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should execute callback after delay', () => {
    const callback = jest.fn();

    renderHook(() => useTimeoutEffect(callback, 1000));

    expect(callback).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(1000);
    });

    expect(callback).toHaveBeenCalledTimes(1);
  });

  it('should not execute callback when delay is null', () => {
    const callback = jest.fn();

    renderHook(() => useTimeoutEffect(callback, null));

    act(() => {
      jest.advanceTimersByTime(10000);
    });

    expect(callback).not.toHaveBeenCalled();
  });

  it('should reset timeout when delay changes', () => {
    const callback = jest.fn();
    let delay = 1000;

    const { rerender } = renderHook(() => useTimeoutEffect(callback, delay));

    act(() => {
      jest.advanceTimersByTime(500);
    });

    delay = 2000;
    rerender();

    act(() => {
      jest.advanceTimersByTime(1500);
    });

    expect(callback).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(callback).toHaveBeenCalledTimes(1);
  });

  it('should clean up on unmount', () => {
    const callback = jest.fn();

    const { unmount } = renderHook(() => useTimeoutEffect(callback, 1000));

    act(() => {
      jest.advanceTimersByTime(500);
    });

    unmount();

    act(() => {
      jest.advanceTimersByTime(1000);
    });

    expect(callback).not.toHaveBeenCalled();
  });
});

describe('useInterval', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should execute callback repeatedly', () => {
    const callback = jest.fn();
    const { result } = renderHook(() => useInterval());

    act(() => {
      result.current.setInterval(callback, 1000);
    });

    act(() => {
      jest.advanceTimersByTime(3500);
    });

    expect(callback).toHaveBeenCalledTimes(3);
  });

  it('should allow clearing interval', () => {
    const callback = jest.fn();
    const { result } = renderHook(() => useInterval());

    let intervalId: NodeJS.Timeout;
    act(() => {
      intervalId = result.current.setInterval(callback, 1000);
    });

    act(() => {
      jest.advanceTimersByTime(2500);
    });

    expect(callback).toHaveBeenCalledTimes(2);

    act(() => {
      result.current.clearInterval(intervalId);
    });

    act(() => {
      jest.advanceTimersByTime(3000);
    });

    expect(callback).toHaveBeenCalledTimes(2);
  });

  it('should clean up all intervals on unmount', () => {
    const callback = jest.fn();
    const { result, unmount } = renderHook(() => useInterval());

    act(() => {
      result.current.setInterval(callback, 1000);
    });

    act(() => {
      jest.advanceTimersByTime(2500);
    });

    expect(callback).toHaveBeenCalledTimes(2);

    unmount();

    act(() => {
      jest.advanceTimersByTime(5000);
    });

    expect(callback).toHaveBeenCalledTimes(2);
  });
});

describe('useIntervalEffect', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should execute callback repeatedly', () => {
    const callback = jest.fn();

    renderHook(() => useIntervalEffect(callback, 1000));

    act(() => {
      jest.advanceTimersByTime(3500);
    });

    expect(callback).toHaveBeenCalledTimes(3);
  });

  it('should not execute callback when delay is null', () => {
    const callback = jest.fn();

    renderHook(() => useIntervalEffect(callback, null));

    act(() => {
      jest.advanceTimersByTime(10000);
    });

    expect(callback).not.toHaveBeenCalled();
  });

  it('should reset interval when delay changes', () => {
    const callback = jest.fn();
    let delay: number | null = 1000;

    const { rerender } = renderHook(() => useIntervalEffect(callback, delay));

    act(() => {
      jest.advanceTimersByTime(2500);
    });

    expect(callback).toHaveBeenCalledTimes(2);

    delay = 500;
    rerender();

    callback.mockClear();

    act(() => {
      jest.advanceTimersByTime(2000);
    });

    expect(callback).toHaveBeenCalledTimes(4);
  });

  it('should clean up on unmount', () => {
    const callback = jest.fn();

    const { unmount } = renderHook(() => useIntervalEffect(callback, 1000));

    act(() => {
      jest.advanceTimersByTime(2500);
    });

    expect(callback).toHaveBeenCalledTimes(2);

    unmount();

    act(() => {
      jest.advanceTimersByTime(5000);
    });

    expect(callback).toHaveBeenCalledTimes(2);
  });
});

describe('useDebounce', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should return initial value immediately', () => {
    const { result } = renderHook(() => useDebounce('initial', 500));

    expect(result.current).toBe('initial');
  });

  it('should debounce value changes', () => {
    let value = 'first';
    const { result, rerender } = renderHook(() => useDebounce(value, 500));

    expect(result.current).toBe('first');

    value = 'second';
    rerender();

    // Value should not have changed yet
    expect(result.current).toBe('first');

    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current).toBe('second');
  });

  it('should reset debounce timer on rapid changes', () => {
    let value = 'a';
    const { result, rerender } = renderHook(() => useDebounce(value, 500));

    value = 'b';
    rerender();

    act(() => {
      jest.advanceTimersByTime(300);
    });

    value = 'c';
    rerender();

    act(() => {
      jest.advanceTimersByTime(300);
    });

    // Still showing 'a' because debounce keeps resetting
    expect(result.current).toBe('a');

    act(() => {
      jest.advanceTimersByTime(200);
    });

    // Now shows 'c' after full debounce period
    expect(result.current).toBe('c');
  });

  it('should clean up timeout on unmount', () => {
    let value = 'initial';
    const { unmount, rerender } = renderHook(() => useDebounce(value, 500));

    value = 'changed';
    rerender();

    unmount();

    // Should not throw after unmount
    expect(() => {
      act(() => {
        jest.advanceTimersByTime(1000);
      });
    }).not.toThrow();
  });
});

describe('useThrottle', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should execute callback immediately on first call', () => {
    const callback = jest.fn();
    const { result } = renderHook(() => useThrottle(callback, 1000));

    act(() => {
      result.current('arg1');
    });

    expect(callback).toHaveBeenCalledTimes(1);
    expect(callback).toHaveBeenCalledWith('arg1');
  });

  it('should throttle subsequent calls', () => {
    const callback = jest.fn();
    const { result } = renderHook(() => useThrottle(callback, 1000));

    act(() => {
      result.current('call1');
      result.current('call2');
      result.current('call3');
    });

    // Only first call should have executed
    expect(callback).toHaveBeenCalledTimes(1);
    expect(callback).toHaveBeenCalledWith('call1');
  });

  it('should execute trailing call after throttle period', () => {
    const callback = jest.fn();
    const { result } = renderHook(() => useThrottle(callback, 1000));

    act(() => {
      result.current('call1');
      result.current('call2');
    });

    expect(callback).toHaveBeenCalledTimes(1);

    act(() => {
      jest.advanceTimersByTime(1000);
    });

    // Trailing call should execute
    expect(callback).toHaveBeenCalledTimes(2);
    expect(callback).toHaveBeenLastCalledWith('call2');
  });

  it('should allow new calls after throttle period', () => {
    const callback = jest.fn();
    const { result } = renderHook(() => useThrottle(callback, 1000));

    act(() => {
      result.current('first');
    });

    act(() => {
      jest.advanceTimersByTime(1100);
    });

    act(() => {
      result.current('second');
    });

    expect(callback).toHaveBeenCalledTimes(2);
    expect(callback).toHaveBeenLastCalledWith('second');
  });

  it('should clean up pending timeout on unmount', () => {
    const callback = jest.fn();
    const { result, unmount } = renderHook(() => useThrottle(callback, 1000));

    act(() => {
      result.current('call1');
      result.current('call2'); // This would be trailing
    });

    unmount();

    // Should not throw and trailing call should not execute
    expect(() => {
      act(() => {
        jest.advanceTimersByTime(2000);
      });
    }).not.toThrow();

    expect(callback).toHaveBeenCalledTimes(1);
  });

  it('should use latest callback when callback reference changes', () => {
    const callback1 = jest.fn();
    const callback2 = jest.fn();
    let callback = callback1;

    const { result, rerender } = renderHook(() => useThrottle(callback, 1000));

    act(() => {
      result.current('arg');
    });

    expect(callback1).toHaveBeenCalledTimes(1);

    callback = callback2;
    rerender();

    act(() => {
      result.current('arg2');
    });

    act(() => {
      jest.advanceTimersByTime(1000);
    });

    // callback2 should be called for the trailing execution
    expect(callback2).toHaveBeenCalledTimes(1);
  });
});
